"""Versioned clean-base QLoRA candidate with portable, verified checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from support.quality_training import prepare
from support.training import (
    adapter_tensor_snapshot, build_transformers_load_kwargs, cuda_memory_snapshot,
    file_sha256, release_gpu_memory, require_cuda_dtype, stable_json_sha256,
    trainable_parameter_digest, verify_lora_only_trainable,
)
from support.training_sessions import latest_complete_checkpoint, load_verified_checkpoint_adapter
from train_dialogue_pilot import (
    PilotProgressCallback, PilotSessionCallback, probe_sampler, reload_smoke,
    select_checkpoint, session_deadline, validate_config, validate_session_seconds,
)
from train_main import atomic_write_json, append_jsonl, build_training_collator, write_progress
from training_control import assert_no_unmanaged_training, gpu_lock, prevent_idle_sleep
from evaluate_quality_candidate import evaluation_session_budget


def checked_run(value: str | Path) -> Path:
    run = (ROOT / value).resolve()
    allowed = (ROOT / 'artifacts/stage5/quality90-v1/training').resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError('quality90 training requires its own candidate directory')
    return run


def validate_quality_config(config):
    validate_config(config, allow_multiple_epochs=True)
    if config.get('initialization') != 'clean_base' or config.get('initial_adapter_path'):
        raise ValueError('this experiment initializes fresh LoRA on the clean base')
    q = config['qlora']
    if type(q['r']) is not int or not 1 <= q['r'] <= 64 or q['bias'] != 'none' or q['target_modules'] != 'all-linear':
        raise ValueError('unsupported LoRA configuration')
    if not 0 < config['training']['learning_rate'] <= 0.001:
        raise ValueError('invalid learning rate')
    checked_run(config['run_dir'])


def train(config, prepared, run, manifest, checkpoint, *, pause_after_steps, seconds, reserve, session_budget):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainerState, TrainingArguments, set_seed

    started = session_budget['measured_at_unix']
    total = config['expected_optimizer_steps']
    initial_step = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))['global_step'] if checkpoint else 0
    manifest['phase'] = 'loading'
    atomic_write_json(run / 'manifest.json', manifest)
    write_progress(run, status='loading', step=initial_step, total=total, started=started)
    append_jsonl(run / 'session_launches.jsonl', {
        'started_at': started, 'hard_deadline_at': started + seconds, 'soft_pause_at': started + seconds - reserve,
        'initial_step': initial_step, 'pause_after_steps': pause_after_steps, 'total_steps': total,
        'resume': str(checkpoint) if checkpoint else None, 'hard_timeout_exit_code': 124,
        'continuous_session_budget': session_budget,
    })
    model = trainer = None
    try:
        dtype = require_cuda_dtype(torch)
        torch.cuda.reset_peak_memory_stats()
        set_seed(config['seed'])
        q = config['qlora']
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                         bnb_4bit_use_double_quant=q['bnb_4bit_use_double_quant'], bnb_4bit_compute_dtype=dtype)
        _, kwargs = build_transformers_load_kwargs(revision=prepared['model_config']['revision'], dtype=dtype,
                                                   quantization_config=quantization)
        model = AutoModelForCausalLM.from_pretrained(str(prepared['model_ref']), **kwargs)
        if config['training'].get('memory_efficient_sdpa'):
            from support.efficient_attention import enable_memory_efficient_sdpa
            atomic_write_json(run / 'attention_backend.json', enable_memory_efficient_sdpa(model))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(
            r=q['r'], lora_alpha=q['lora_alpha'], lora_dropout=q['lora_dropout'],
            bias='none', task_type='CAUSAL_LM', target_modules='all-linear'))
        trainability = verify_lora_only_trainable(model)
        initial_digest = trainable_parameter_digest(model)
        manifest['initialization'] = 'fresh_lora_on_clean_base'
        atomic_write_json(run / 'manifest.json', manifest)
        tc = config['training']
        tokenizer = prepared['tokenizer']
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        args = TrainingArguments(
            output_dir=str(run / '_trainer'), max_steps=total, num_train_epochs=tc['num_train_epochs'],
            per_device_train_batch_size=tc['per_device_train_batch_size'], gradient_accumulation_steps=tc['gradient_accumulation_steps'],
            learning_rate=tc['learning_rate'], lr_scheduler_type=tc['lr_scheduler_type'], warmup_ratio=tc['warmup_ratio'], optim=tc['optim'],
            logging_steps=1, save_strategy='steps', save_steps=1, save_total_limit=3,
            dataloader_num_workers=0, remove_unused_columns=False, gradient_checkpointing=True,
            bf16=dtype == torch.bfloat16, fp16=dtype != torch.bfloat16, seed=config['seed'], report_to=[],
            torch_empty_cache_steps=tc.get('torch_empty_cache_steps'), group_by_length=True, logging_nan_inf_filter=False,
        )
        portable = PilotSessionCallback(output_dir=run, total_steps=total, enabled=True,
                                        resumed_checkpoint=str(checkpoint) if checkpoint else None,
                                        session_seconds=seconds-reserve, pause_after_steps=pause_after_steps, initial_step=initial_step)
        portable.started_at = started
        trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(prepared['features']),
                          data_collator=build_training_collator(tokenizer, config, prepared['model_config']),
                          callbacks=[PilotProgressCallback(run, started, total,
                              clear_cache_at_microbatch_boundary=tc.get('empty_cache_at_microbatch_boundary', False)), portable])
        atomic_write_json(run / 'sampler_probe.json', probe_sampler(trainer, prepared))
        write_progress(run, status='training', step=initial_step, total=total, started=started)
        if checkpoint and initial_step == total:
            load_verified_checkpoint_adapter(model, checkpoint, device=str(args.device))
            trainer.state = TrainerState.load_from_json(str(checkpoint / 'trainer_state.json'))
        else:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        step = int(trainer.state.global_step)
        losses = [float(row['loss']) for row in trainer.state.log_history if 'loss' in row]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise RuntimeError('training must report finite loss')
        last = latest_complete_checkpoint(run, expected_total_steps=total)
        if last is None or int(last.name.rsplit('-', 1)[1]) != step:
            raise RuntimeError('training ended without a verified checkpoint at the actual step')
        if step < total:
            if not portable.pause_requested:
                raise RuntimeError('unexpected early stop')
            manifest.update(phase='paused', global_step=step, latest_checkpoint=str(last))
            atomic_write_json(run / 'manifest.json', manifest)
            write_progress(run, status='paused', step=step, total=total, started=started,
                           loss=losses[-1], latest_complete_checkpoint=str(last))
            return 0
        if step != total:
            raise RuntimeError('optimizer horizon exceeded')
        final_digest = trainable_parameter_digest(model)
        if final_digest == initial_digest:
            raise RuntimeError('adapter weights did not change')
        expected = adapter_tensor_snapshot(model)
        peak = cuda_memory_snapshot(torch)
        adapter = run / 'final_adapter'
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        trainer = model = None
        release_gpu_memory(torch)
        generation = reload_smoke(prepared['model_config'], adapter, expected, prepared['reload_example'])
        atomic_write_json(run / 'training_report.json', {
            'artifact_status': config['artifact_status'], 'optimizer_steps': step, 'losses': losses,
            'initialization': 'fresh_lora_on_clean_base', 'initial_digest': initial_digest, 'final_digest': final_digest,
            'lora_trainability': trainability, 'training_memory': peak, 'generation_check': generation, 'promotion': False})
        manifest.update(phase='completed', global_step=step, final_adapter=str(adapter), promotion=False)
        atomic_write_json(run / 'manifest.json', manifest)
        write_progress(run, status='completed', step=step, total=total, started=started, adapter_path=str(adapter))
        return 0
    except BaseException as exc:
        manifest.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(run / 'manifest.json', manifest)
        previous = json.loads((run / 'progress.json').read_text(encoding='utf-8')) if (run / 'progress.json').exists() else {}
        write_progress(run, status='failed', step=int(previous.get('step', initial_step)), total=total,
                       started=started, error=manifest['error'])
        raise
    finally:
        trainer = model = None
        release_gpu_memory(torch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/train-quality90-v1.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--allow-pending-review', action='store_true')
    parser.add_argument('--resume', nargs='?', const='latest')
    parser.add_argument('--pause-after-steps', type=int)
    parser.add_argument('--session-seconds', type=float)
    parser.add_argument('--pause-reserve-seconds', type=float)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_quality_config(config)
    run = checked_run(args.run_dir or config['run_dir'])
    seconds = args.session_seconds if args.session_seconds is not None else config['session_seconds']
    reserve = args.pause_reserve_seconds if args.pause_reserve_seconds is not None else config['pause_reserve_seconds']
    validate_session_seconds(seconds, reserve)
    if args.pause_after_steps is not None and args.pause_after_steps <= 0:
        raise ValueError('pause-after-steps must be positive')
    if args.preflight_only and args.resume:
        raise ValueError('preflight-only cannot resume')
    # The controller alone may release a manual pause, with explicit resume.
    if not args.preflight_only and ((run / 'USER_PAUSED.json').exists()
                                   or (ROOT / 'artifacts/stage5/quality90-v1/USER_PAUSED.json').exists()):
        raise ValueError('manual pause is latched; use the explicit resume controller')
    with gpu_lock(run / '.run.lock'):
        path = run / 'manifest.json'
        existing = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if not existing and ((run / '_trainer').exists() or (run / 'final_adapter').exists()):
            raise ValueError('refusing to overwrite untracked training artifacts')
        prepared = prepare(config, allow_pending_review=args.allow_pending_review)
        identity = {'schema': 'quality90-run-v1', 'config': config, 'config_sha256': file_sha256(config_path),
                    'integrity': prepared['integrity_manifest']}
        fingerprint = stable_json_sha256(identity)
        if existing:
            if existing['fingerprint'] != fingerprint or existing['identity'] != identity:
                raise ValueError('run inputs changed; create a new candidate, do not resume it')
            if not args.preflight_only and not args.resume and existing['phase'] != 'preflight_only':
                raise ValueError('existing run requires explicit resume')
            if existing['phase'] == 'completed' and not args.preflight_only:
                raise ValueError('this run is already completed')
        elif args.resume:
            raise ValueError('resume requires an existing run')
        checkpoint = select_checkpoint(run, args.resume, config['expected_optimizer_steps'])
        manifest = existing or {'identity': identity, 'fingerprint': fingerprint, 'phase': 'preflight_only', 'promotion': False}
        atomic_write_json(run / 'preflight.json', prepared['integrity_manifest'])
        if not existing:
            atomic_write_json(path, manifest)
        if args.preflight_only:
            print(json.dumps({'status': 'preflight_pass', 'train_targets': len(prepared['features']),
                              'fingerprint': fingerprint, 'run_dir': str(run)}))
            return 0
        with gpu_lock(ROOT / 'artifacts/stage4/portable-gpu.lock'), prevent_idle_sleep():
            from live_demo import assert_no_legacy_inference
            from benchmark_live_runtime import _assert_no_live_server_process
            assert_no_unmanaged_training()
            assert_no_legacy_inference()
            _assert_no_live_server_process()
            # Reuse the frozen evaluator's window validation, without reading cases.
            # Recompute after preflight/process checks so restarts cannot reset time.
            budget = evaluation_session_budget(seconds / 3600)
            seconds = budget['hard_seconds']
            soft_seconds = min(budget['soft_seconds'], seconds - reserve)
            if soft_seconds <= 0:
                raise ValueError('Session budget exhausted; preserve the checkpoint until explicit continuation')
            budget['soft_seconds'] = soft_seconds
            budget['soft_stop_at_unix'] = budget['measured_at_unix'] + soft_seconds
            reserve = seconds - soft_seconds
            with session_deadline(seconds):
                return train(config, prepared, run, manifest, checkpoint,
                             pause_after_steps=args.pause_after_steps, seconds=seconds,
                             reserve=reserve, session_budget=budget)


if __name__ == '__main__':
    raise SystemExit(main())
