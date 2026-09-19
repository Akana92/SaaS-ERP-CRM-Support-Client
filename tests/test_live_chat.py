from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support.live_chat import generate_chat_for_runner
from support.live_precision import LocalPrecisionRunner


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (1, len(self.data[0]))

    def to(self, _device):
        return self

    def new_ones(self, shape):
        return FakeTensor([[1] * shape[-1]])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            row = self.data[0] if isinstance(rows, slice) else self.data[rows]
            if isinstance(cols, slice):
                return FakeTensor([row[cols]])
        return self.data[key]


class ChatTokenizer:
    eos_token_id = 99

    def __init__(self, *, fail=False, bad_json=False):
        self.fail = fail
        self.bad_json = bad_json
        self.messages = []

    def apply_chat_template(self, messages, *, add_generation_prompt, return_tensors):
        if self.fail:
            raise RuntimeError("tokenizer failed")
        self.messages.append(messages)
        return FakeTensor([[10, 11, 12]])

    def decode(self, token_ids, *, skip_special_tokens):
        if self.bad_json:
            return "not json"
        return json.dumps(
            {
                "category": "Payment",
                "priority": "High",
                "sentiment": "Neutral",
                "recommended_action": "check_payment",
                "suggested_response": "Оплата подтверждена.",
                "human_escalation": False,
                "escalation_reason": None,
                "evidence_ids": ["erp.invoice.status", "policy.response.minimal"],
            },
            ensure_ascii=False,
        )


class ChatModel:
    device = "cuda:0"

    def __init__(self, *, fail=False):
        self.fail = fail
        self.generate_kwargs = []

    def generate(self, input_ids, **kwargs):
        self.generate_kwargs.append(kwargs)
        if self.fail:
            raise RuntimeError("generate failed")
        return FakeTensor([input_ids.data[0] + [21, 22, 99]])


def model_config():
    return {
        "key": "qwen3_4b",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "c" * 40,
        "local_path": "models/qwen3_4b",
    }


def model_input():
    return {
        "customer_message": "Счёт оплачен?",
        "erp_context": {"source_status": "ok", "facts": {"erp.invoice.status": "paid"}},
        "policy_version": "demo-v1",
        "policy_rules": [{"id": "policy.response.minimal", "text": "Отвечать кратко."}],
    }


def loaded_runner(*, tokenizer=None, model=None):
    runner = LocalPrecisionRunner(model_config(), adapter_path="adapter", inference_profile="nf4")
    runner._tokenizer = tokenizer or ChatTokenizer()
    runner._model = model or ChatModel()
    runner._loaded = True
    runner._generation_lock = threading.Lock()
    return runner


class LiveChatTests(unittest.TestCase):
    def test_generate_chat_uses_actual_messages_for_tokenization_and_original_input_for_validation(self):
        tokenizer = ChatTokenizer()
        model = ChatModel()
        runner = loaded_runner(tokenizer=tokenizer, model=model)
        chat_messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "current"}]

        run = generate_chat_for_runner(runner, chat_messages, model_input(), "req-chat", max_new_tokens=64)

        self.assertIsNone(run.error)
        self.assertEqual(tokenizer.messages, [chat_messages])
        self.assertEqual(model.generate_kwargs[0]["max_new_tokens"], 64)
        self.assertEqual(model.generate_kwargs[0]["logits_to_keep"], 1)
        self.assertEqual(run.usage.input_tokens, 3)
        self.assertEqual(run.usage.output_tokens, 3)
        self.assertEqual(run.result.category, "Payment")

    def test_generate_chat_returns_incomplete_usage_on_tokenizer_error(self):
        runner = loaded_runner(tokenizer=ChatTokenizer(fail=True))

        run = generate_chat_for_runner(runner, [], model_input(), "req-tokenizer")

        self.assertIn("tokenizer failed", run.error)
        self.assertFalse(run.usage.complete)
        self.assertIsNone(run.usage.input_tokens)

    def test_generate_chat_keeps_known_input_usage_on_generation_error(self):
        runner = loaded_runner(model=ChatModel(fail=True))

        run = generate_chat_for_runner(runner, [{"role": "user", "content": "u"}], model_input(), "req-generate")

        self.assertIn("generate failed", run.error)
        self.assertFalse(run.usage.complete)
        self.assertEqual(run.usage.input_tokens, 3)

    def test_generate_chat_reports_parser_error_with_complete_usage(self):
        runner = loaded_runner(tokenizer=ChatTokenizer(bad_json=True))

        run = generate_chat_for_runner(runner, [{"role": "user", "content": "u"}], model_input(), "req-parse")

        self.assertIn("invalid JSON", run.error)
        self.assertTrue(run.usage.complete)
        self.assertIsNone(run.result)

    def test_unknown_evidence_is_an_output_error_not_runtime_unavailability(self):
        source = model_input()
        source["erp_context"]["facts"] = {}
        run = generate_chat_for_runner(loaded_runner(), [{"role": "user", "content": "u"}], source, "req-evidence")
        self.assertTrue(run.error.startswith("model_output_invalid:"))
        self.assertTrue(run.usage.complete)
        self.assertIsNone(run.result)
        self.assertIn("erp.invoice.status", run.raw_text)


if __name__ == "__main__":
    unittest.main()
