"""Download public, pinned model snapshots into this workspace only.

Default mode uses Hugging Face snapshot_download. Use --ranged only as a fallback
when full-file streaming stalls but public HTTP Range requests are healthy.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"

CHUNK_BYTES = 16 * 1024 * 1024
CHUNK_WORKERS = 4
CHUNK_RETRIES = 2
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


@dataclass(frozen=True)
class WeightFile:
    path: str
    size: int
    sha256: str


def load_selected_configs(config_path: Path, model_key: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = [item for item in config["candidates"] if not model_key or item["key"] == model_key]
    if model_key and not selected:
        raise ValueError(f"unknown model key: {model_key}")
    return config, selected


def resolve_metadata(item: dict[str, Any]):
    from huggingface_hub import HfApi

    api = HfApi(token=False)
    revision = item.get("revision") or "main"
    info = api.model_info(item["model_id"], revision=revision, files_metadata=True, token=False)
    if not info.sha:
        raise ValueError(f"could not resolve exact revision for {item['model_id']}")
    item["revision"] = info.sha
    item["source"] = f"https://huggingface.co/{item['model_id']}/tree/{info.sha}"
    item["license"] = (info.card_data or {}).get("license", "unknown")
    return info


def write_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normal_snapshot_download(item: dict[str, Any], metadata_only: bool) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    local = validate_model_destination(item)
    if metadata_only:
        allow_patterns = ["*.json", "*.model", "merges.txt", "vocab.json", "LICENSE*", "README.md"]
    else:
        allow_patterns = ["*.json", "*.safetensors", "*.model", "merges.txt", "vocab.json", "LICENSE*", "README.md"]
    snapshot_download(
        repo_id=item["model_id"],
        revision=item["revision"],
        token=False,
        local_dir=local,
        cache_dir=ROOT / ".cache/huggingface/hub",
        allow_patterns=allow_patterns,
        max_workers=2,
    )
    return save_provenance(item, local, "snapshot_download", metadata_only=metadata_only, weights=[])


def ranged_snapshot_download(item: dict[str, Any], info) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    local = validate_model_destination(item)
    snapshot_download(
        repo_id=item["model_id"],
        revision=item["revision"],
        token=False,
        local_dir=local,
        cache_dir=ROOT / ".cache/huggingface/hub",
        allow_patterns=["*.json", "*.model", "merges.txt", "vocab.json", "LICENSE*", "README.md"],
        ignore_patterns=["*.safetensors"],
        max_workers=2,
    )
    weights = weight_files_from_metadata(info)
    downloaded = []
    for weight in weights:
        downloaded.append(download_weight_ranged(item, weight, local))
    return save_provenance(item, local, "ranged", metadata_only=False, weights=downloaded)


def weight_files_from_metadata(info) -> list[WeightFile]:
    weights = []
    for sibling in getattr(info, "siblings", []) or []:
        path = getattr(sibling, "rfilename", None)
        if not isinstance(path, str) or not path.endswith(".safetensors"):
            continue
        lfs = getattr(sibling, "lfs", None) or {}
        sha256 = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size") or getattr(sibling, "size", None)
        if not sha256 or size is None:
            raise ValueError(f"missing LFS metadata for weight file {path}")
        weights.append(WeightFile(path=path, size=int(size), sha256=str(sha256)))
    if not weights:
        raise ValueError("no safetensors LFS weight files found in metadata")
    return weights


def download_weight_ranged(item: dict[str, Any], weight: WeightFile, local_root: Path) -> dict[str, Any]:
    target = local_root / weight.path
    if target.exists() and target.stat().st_size == weight.size and sha256_file(target) == weight.sha256:
        return {"path": weight.path, "bytes": weight.size, "sha256": weight.sha256, "status": "already_present"}
    part_dir = ROOT / ".cache" / "ranged_downloads" / item["key"] / item["revision"] / weight.path
    part_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(chunk_ranges(weight.size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as executor:
        futures = [
            executor.submit(download_chunk, item["model_id"], item["revision"], weight.path, start, end, weight.size, part_dir)
            for start, end in chunks
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent, prefix=target.name + ".", suffix=".tmp") as tmp:
        temp_path = Path(tmp.name)
        for start, end in chunks:
            part = chunk_part_path(part_dir, start, end)
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"wrong cached chunk size for {weight.path} bytes {start}-{end}")
            with part.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    tmp.write(block)
    if temp_path.stat().st_size != weight.size:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"assembled size mismatch for {weight.path}")
    digest = sha256_file(temp_path)
    if digest != weight.sha256:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {weight.path}")
    os.replace(temp_path, target)
    return {"path": weight.path, "bytes": weight.size, "sha256": digest, "status": "downloaded"}


def download_chunk(
    repo_id: str,
    revision: str,
    file_path: str,
    start: int,
    end: int,
    total_size: int,
    part_dir: Path,
) -> None:
    expected_size = end - start + 1
    part_path = chunk_part_path(part_dir, start, end)
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return
    url = hf_resolve_url(repo_id, revision, file_path)
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    last_error = None
    for attempt in range(CHUNK_RETRIES + 1):
        try:
            stream_chunk(url, headers, part_path, start, end, total_size, expected_size)
            return
        except Exception as exc:
            last_error = exc
            if attempt < CHUNK_RETRIES:
                time.sleep(1 + attempt)
    raise RuntimeError(f"failed chunk {file_path} bytes {start}-{end}: {last_error}") from last_error


def stream_chunk(
    url: str,
    headers: dict[str, str],
    part_path: Path,
    start: int,
    end: int,
    total_size: int,
    expected_size: int,
) -> None:
    import requests

    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(20, 120)) as response:
        if response.status_code != 206:
            raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
        content_range = response.headers.get("Content-Range", "")
        match = CONTENT_RANGE_RE.match(content_range)
        if not match:
            raise RuntimeError("missing or invalid Content-Range")
        got_start, got_end, got_total = (int(value) for value in match.groups())
        if (got_start, got_end, got_total) != (start, end, total_size):
            raise RuntimeError("Content-Range does not match requested chunk")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected_size:
            raise RuntimeError("Content-Length does not match requested chunk")
        tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if tmp_path.stat().st_size != expected_size:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("downloaded chunk size mismatch")
        os.replace(tmp_path, part_path)


def save_provenance(
    item: dict[str, Any],
    local: Path,
    method: str,
    *,
    metadata_only: bool,
    weights: list[dict[str, Any]],
) -> dict[str, Any]:
    files = [
        {"path": str(path.relative_to(local)).replace("\\", "/"), "bytes": path.stat().st_size}
        for path in local.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    ]
    report = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "metadata_only": metadata_only,
        "model_key": item["key"],
        "model_id": item["model_id"],
        "revision": item["revision"],
        "source": item.get("source"),
        "license": item.get("license"),
        "local_path": item["local_path"],
        "files": files,
        "weights": weights,
        "total_bytes": sum(file["bytes"] for file in files),
    }
    dest = ROOT / "artifacts/stage2" / f"{item['key']}-source.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_model_destination(item: dict[str, Any]) -> Path:
    local = (ROOT / item["local_path"]).resolve()
    models_root = (ROOT / "models").resolve()
    if not local.is_relative_to(models_root):
        raise ValueError("model destination must stay in workspace models/")
    return local


def chunk_ranges(size: int):
    start = 0
    while start < size:
        end = min(start + CHUNK_BYTES - 1, size - 1)
        yield start, end
        start = end + 1


def chunk_part_path(part_dir: Path, start: int, end: int) -> Path:
    return part_dir / f"{start:016d}-{end:016d}.part"


def hf_resolve_url(repo_id: str, revision: str, file_path: str) -> str:
    encoded_repo = quote(repo_id, safe="/")
    encoded_revision = quote(revision, safe="")
    encoded_path = quote(file_path, safe="/")
    return f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_path}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=["qwen3_4b", "qwen25_15b"])
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--ranged", action="store_true", help="Use verified public HTTP Range GETs for safetensors files")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/models.json")
    args = parser.parse_args()
    config, selected = load_selected_configs(args.config, args.model_key)
    metadata_by_key = {}
    for item in selected:
        metadata_by_key[item["key"]] = resolve_metadata(item)
    write_config(args.config, config)
    for item in selected:
        print(
            json.dumps(
                {
                    "model": item["model_id"],
                    "revision": item["revision"],
                    "metadata_only": args.metadata_only,
                    "ranged": args.ranged,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.metadata_only:
            report = normal_snapshot_download(item, metadata_only=True)
        elif args.ranged:
            report = ranged_snapshot_download(item, metadata_by_key[item["key"]])
        else:
            report = normal_snapshot_download(item, metadata_only=False)
        print(
            json.dumps(
                {"download_complete": item["key"], "method": report["method"], "total_bytes": report["total_bytes"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
