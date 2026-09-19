"""Train-only input boundary for the independent quality90 experiment series.

No validation/test loader or path is accepted here. Their contents are never
tokenized by this module. Dataset and implementation hashes pin every resume.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
import os
import re
from pathlib import Path
from typing import Any

from support.dialogue_dataset import (
    Dialogue, PreparedTurn, encode_prepared_turn, load_dialogue_policy,
    load_dialogues, prepare_turn, public_history,
)
from support.contracts import PolicyDocument
from support.live_dialogue import TokenCounter, compose_dialogue_messages
from support.training import file_sha256, load_model_config, package_versions, stable_json_sha256

ROOT = Path(__file__).resolve().parents[2]


def training_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    allowed = (ROOT / 'data/quality90_v1/train').resolve()
    if path.parent != allowed or path.suffix != '.jsonl':
        raise ValueError('only explicit quality90 train JSONL files may supply training features')
    return path


def bound_training_paths(config: dict[str, Any]) -> list[Path]:
    sources = config.get('train_sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('train_sources must contain frozen training file hashes')
    # Validate every path before opening even the first file.
    paths = [training_path(name) for name in sources]
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate resolved training paths')
    for path, expected in zip(paths, sources.values()):
        if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
            raise ValueError(f'training source hash changed: {path.name}')
    return paths


def isolation_manifest_path(config: dict[str, Any]) -> Path:
    value = config.get('isolation_manifest', 'data/quality90_v1/isolation_manifest.json')
    if not isinstance(value, (str, Path)):
        raise ValueError('isolation_manifest must name a local isolation receipt')
    path = (ROOT / value).resolve()
    if (path.parent != (ROOT / 'data/quality90_v1').resolve()
            or not re.fullmatch(r'isolation_manifest(?:_[A-Za-z0-9_-]+)?\.json', path.name)):
        raise ValueError('isolation_manifest must be a direct quality90 isolation_manifest JSON file')
    return path


def validate_isolation(config: dict[str, Any]) -> dict[str, Any]:
    path = isolation_manifest_path(config)
    if file_sha256(path) != config.get('isolation_manifest_sha256'):
        raise ValueError('isolation receipt hash changed or was not pinned')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('schema') != 'quality90-split-isolation-v1' or receipt.get('status') != 'pass':
        raise ValueError('training requires a passing split isolation audit')
    if receipt.get('train_sources') != config['train_sources']:
        raise ValueError('isolation audit belongs to different training data')
    if receipt.get('check_summary', {}).get('unresolved_train_ids') != []:
        raise ValueError('unresolved cross-split training overlap')
    return receipt


def validate_train_identities(rows: list[dict[str, Any]]) -> None:
    ids = set()
    for row in rows:
        if row['split'] != 'pilot_train' or not row['family_id'].startswith('train-'):
            raise ValueError('training split/family boundary violated')
        if row['id'] in ids:
            raise ValueError('duplicate training dialogue')
        ids.add(row['id'])


def check_completion(feature: dict[str, list[int]], eos: int) -> None:
    tokens, labels, attention = (feature[key] for key in ('input_ids', 'labels', 'attention_mask'))
    if not tokens or len(tokens) != len(labels) or len(tokens) != len(attention):
        raise ValueError('unaligned tensors')
    start = next((i for i, label in enumerate(labels) if label != -100), len(labels))
    end = max((i for i, label in enumerate(labels) if label != -100), default=-1)
    if start == 0 or start == len(labels) or labels[end] != eos:
        raise ValueError('prompt mask or supervised EOS invalid')
    # Chat templates may leave ignored whitespace after the supervised EOS.
    if labels[start:end+1] != tokens[start:end+1] or any(v != 1 for v in attention):
        raise ValueError('only the entire current completion may be supervised')


def validate_training_history_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in ('public_gold', 'raw_gold'):
        raise ValueError('training_history_mode must be public_gold or raw_gold')
    return value


def prepare_training_turn(
    dialogue: Dialogue, turn_index: int, policy: PolicyDocument, token_counter: TokenCounter,
    *, training_history_mode: str = 'public_gold',
) -> tuple[PreparedTurn, int]:
    """Prepare controlled train history; count actual response changes vs public gold.

    raw_gold is a teacher-forced training experiment, not observed runtime history.
    Historical structured labels are never supplied to the shared renderer.
    """
    mode = validate_training_history_mode(training_history_mode)
    if dialogue.split != 'pilot_train':
        raise ValueError('training history requires pilot_train')
    prepared = prepare_turn(dialogue, turn_index, policy, token_counter)
    if mode == 'public_gold':
        return prepared, 0
    history = public_history(dialogue, turn_index)
    changed = 0
    for previous, turn in zip(history, dialogue.turns[:turn_index]):
        response = turn.expected.suggested_response
        if previous['client']['response'] != response:
            previous['client']['response'] = response
            changed += 1
    if changed:
        messages, metadata = compose_dialogue_messages(prepared.model_input, history, token_counter)
        prepared = replace(prepared, chat_messages=messages, metadata=metadata)
    return prepared, changed


def prepare(config: dict[str, Any], *, allow_pending_review: bool) -> dict[str, Any]:
    history_mode = validate_training_history_mode(config.get('training_history_mode', 'public_gold'))
    paths = bound_training_paths(config)
    validate_isolation(config)
    dialogues = [item for path in paths for item in load_dialogues(path, 'pilot_train')]
    validate_train_identities([d.model_dump() for d in dialogues])
    if len(dialogues) != config['expected_dialogues']:
        raise ValueError('unexpected training dialogue count')
    if not allow_pending_review and any(d.review_status != 'human_approved' for d in dialogues):
        raise ValueError('draft training requires --allow-pending-review')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model_config = load_model_config(ROOT / config['model_config'], config['model_key'])
    model_ref = (ROOT / model_config['local_path']).resolve()
    if not model_ref.is_relative_to((ROOT / 'models').resolve()):
        raise ValueError('local model must be in the project model directory')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_ref), local_files_only=True, trust_remote_code=False)
    policy = load_dialogue_policy()

    def count(messages):
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    features, records, probes = [], [], []
    for dialogue in dialogues:
        for index in range(len(dialogue.turns)):
            prepared, history_changes = prepare_training_turn(
                dialogue, index, policy, count, training_history_mode=history_mode)
            if prepared.metadata.omitted_turn_ids:
                raise ValueError(f'history omitted: {prepared.case_id}')
            encoded = encode_prepared_turn(prepared, tokenizer, max_length=config['max_length'])
            feature = encoded.to_features()
            clean = {key: feature[key] for key in ('input_ids', 'labels', 'attention_mask')}
            check_completion(clean, encoded.mask_probe.terminal_eos_token_id)
            features.append(clean)
            records.append({'case_id': prepared.case_id, 'chat_messages': prepared.chat_messages,
                            'model_input': prepared.model_input, 'target_json': prepared.target_json,
                            'history_turns': index, 'training_history_mode': history_mode,
                            'history_responses_changed': history_changes})
            probes.append(asdict(encoded.mask_probe))
    if len(features) != config['expected_train_rows']:
        raise ValueError('unexpected training target count')
    source_paths = paths + [isolation_manifest_path(config)] + [ROOT / name for name in (
        config['model_config'], 'data/policy/employee-telecom-v3.json',
        'scripts/train_quality_candidate.py', 'src/support/quality_training.py',
        'scripts/evaluate_quality_candidate.py',
        'scripts/train_dialogue_pilot.py', 'scripts/train_main.py',
        'scripts/training_control.py', 'src/support/training_sessions.py',
        'src/support/training.py', 'src/support/completion_logits.py',
        'src/support/efficient_attention.py', 'src/support/contracts.py',
        'src/support/dialogue_dataset.py', 'src/support/live_dialogue.py',
        'src/support/prompting.py', 'src/support/graph.py', 'src/support/modeling.py',
        'src/support/live_precision.py')]
    source_paths += [p for p in model_ref.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.jinja'}]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): file_sha256(p) for p in source_paths}
    # Reject data edits while tokenization was running.
    bound_training_paths(config)
    validate_isolation(config)
    categories = Counter(t.expected.category for d in dialogues for t in d.turns)
    summary = {'schema': 'quality90-training-integrity-v1', 'train_dialogues': len(dialogues),
               'training_history_mode': history_mode,
               'history_responses_changed': sum(r['history_responses_changed'] for r in records),
               'targets_with_history_changes': sum(r['history_responses_changed'] > 0 for r in records),
               'train_targets': len(features), 'families': len({d.family_id for d in dialogues}),
               'categories': dict(categories), 'max_tokens': max(len(f['input_ids']) for f in features),
               'training_tokens': sum(len(f['input_ids']) for f in features),
               'supervised_tokens': sum(sum(v != -100 for v in f['labels']) for f in features),
               'encoded_sha256': stable_json_sha256(features), 'source_hashes': hashes,
               'policy_sha256': stable_json_sha256(policy.model_dump(mode='json')),
               'packages': package_versions(['torch', 'transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate']),
               'holdout_read_or_encoded': False, 'history_truncations': 0,
               'initialization': 'fresh_lora_on_clean_base', 'human_review': 'pending', 'promotion': False}
    return {'features': features, 'records': records, 'probes': probes, 'model_config': model_config,
            'model_ref': model_ref, 'tokenizer': tokenizer, 'source_hashes': hashes,
            'integrity_manifest': summary, 'reload_example': next(r for r in records if r['history_turns'] > 0)}
