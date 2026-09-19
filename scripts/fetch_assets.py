"""Fetch pinned, allowlisted inference assets and optional TRAIN data.

Existing files are checked before network access and never replaced. All assets
have pinned SHA-256 checks; the base model index is also structurally checked.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/asset-sources.json"


class AssetError(Exception):
    """Safe user-facing message; never include upstream exception text."""


def safe_failure(exc: Exception) -> str:
    """Classify wrapped failures without echoing URLs, paths or credentials."""
    pending, seen, failures = [exc], set(), []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        failures.append(current)
        for nested in (current.__cause__, current.__context__, getattr(current, "reason", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    for current in failures:
        if getattr(current, "errno", None) == errno.ENOSPC or getattr(current, "winerror", None) in (39, 112):
            return "Disk full; free space on the temporary and destination drives, then retry."
    for current in failures:
        status = getattr(getattr(current, "response", None), "status_code", None)
        if type(status) is int and status in (401, 403):
            return f"Repository access denied (HTTP {status}); check access and token secret."
    for current in failures:
        if isinstance(current, (ConnectionError, TimeoutError)) or any(
            kind.__name__ in ("ConnectionError", "Timeout", "SSLError", "URLError")
            for kind in type(current).__mro__
        ):
            code = getattr(current, "errno", None)
            detail = f" (errno {code})" if type(code) is int else ""
            return f"Connection/network failure{detail}; check connectivity and certificate trust."
    if isinstance(exc, OSError):
        detail = f" (errno {exc.errno})" if type(exc.errno) is int else ""
        return f"Filesystem failure{detail}; check available space and permissions."
    return "Download failed; check network, repository access and token secret."


def safe_path(root: Path, name: str = "") -> Path:
    relative = PurePosixPath(name)
    if name and (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name):
        raise AssetError("Unsafe asset path in configuration.")
    target = root.absolute() / relative
    for part in (target, *target.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise AssetError("Asset destinations must not contain symbolic links or junctions.")
    if not target.resolve().is_relative_to(root.resolve()):
        raise AssetError("Asset path escapes destination.")
    return target


def valid_file(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size == 0:
        raise AssetError("Existing asset is not a nonempty file; move it aside before retrying.")
    if "bytes" in expected and path.stat().st_size != expected["bytes"]:
        raise AssetError("Existing asset size differs from approved export; no files overwritten.")
    if "sha256" in expected:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise AssetError("Asset SHA-256 differs from approved export; no files overwritten.")
    return True


def verify_structure(key: str, destination: Path, spec: dict) -> None:
    if key != "base":
        return
    try:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not isinstance(json.loads(safe_path(destination, name).read_text(encoding="utf-8")), dict):
                raise ValueError()
        index = json.loads(safe_path(destination, "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        expected = {name for name in spec["files"] if name.endswith(".safetensors")}
        if shards != expected:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise AssetError("Base model JSON/index is invalid or references unexpected shards.") from None


def fetch_asset(key: str, spec: dict, destination: Path, *, token=None,
                verify_only=False, downloader=None) -> str:
    files = spec["files"]
    if not files:
        raise AssetError("Asset allowlist is empty.")
    if key == "dataset" and any(not name.startswith("data/quality90_v1/train/") or
                                not name.endswith(".jsonl") for name in files):
        raise AssetError("Dataset allowlist must contain TRAIN JSONL files only.")
    missing = [name for name, expected in files.items()
               if not valid_file(safe_path(destination, name), expected)]
    if not missing:
        verify_structure(key, destination, spec)
        return "verified local files"
    if verify_only:
        raise AssetError(f"{key}: {len(missing)} required files missing; run without --verify-only.")
    revision = spec.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AssetError(f"{key}: published commit revision is not pinned in configs/asset-sources.json.")
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    # Stage away from the project. Never allow HF cards/cache metadata to replace
    # application files, and verify every requested file before installing any.
    with tempfile.TemporaryDirectory(prefix="support-assets-") as staging:
        stage = Path(staging)
        try:
            downloader(repo_id=spec["repo_id"], repo_type=spec["repo_type"],
                       revision=revision, allow_patterns=missing, local_dir=stage,
                       token=token, max_workers=2)
        except Exception as exc:
            raise AssetError(f"{key}: {safe_failure(exc)}") from None
        for name in missing:
            if not valid_file(safe_path(stage, name), files[name]):
                raise AssetError(f"{key}: download did not contain all required files.")
        for name in missing:
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create also guards against replacing a concurrently written file.
            with target.open("xb") as output, safe_path(stage, name).open("rb") as source:
                try:
                    shutil.copyfileobj(source, output)
                except BaseException:
                    output.close()
                    target.unlink()
                    raise
    verify_structure(key, destination, spec)
    return "downloaded and verified"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "models/qwen3_4b")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "adapters/quality-f")
    parser.add_argument("--dataset", action="store_true", help="Also fetch optional TRAIN only")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if sys.platform == "win32" and not args.verify_only:
            import truststore
            truststore.inject_into_ssl()
        token = None
        if args.token_file and args.token_file.exists():
            token = args.token_file.read_text(encoding="utf-8").strip() or None
        specs = json.loads(CONFIG.read_text(encoding="utf-8"))
        selections = [("base", args.base_dir), ("adapter", args.adapter_dir)]
        if args.dataset:
            selections.append(("dataset", args.dataset_dir))
        for key, destination in selections:
            result = fetch_asset(key, specs[key], destination, token=token, verify_only=args.verify_only)
            print(f"{key}: {result} ({specs[key]['verification']})")
        return 0
    except AssetError as exc:
        print(f"Asset setup failed: {exc}", file=sys.stderr)
    except OSError as exc:
        print(f"Asset setup failed: {safe_failure(exc)}", file=sys.stderr)
    except Exception:
        print("Asset setup failed: check configuration, dependency installation and filesystem permissions.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
