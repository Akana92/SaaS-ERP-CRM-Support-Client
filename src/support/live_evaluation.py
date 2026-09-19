"""Read only the pinned, descriptive candidate F aggregate for the local demo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = ROOT / "artifacts/stage5/quality90-v1/reports/candidate-f-semantic-round06/aggregate-summary.json"
VERIFICATION = ROOT / "artifacts/stage5/quality90-v1/verification/candidate-f-final-assessment-verification.json"
CANDIDATE_REPORT = ROOT / "docs/CANDIDATE_F_RESULTS.md"
AGGREGATE_SHA256 = "762c469663f92d04e6fce59e14886c7eb1e6aa85634b89a9b9d51e544e344318"
VERIFICATION_SHA256 = "df7626349a9afc10810f843d22ba46b6ee07affa30bcd88de6ad91abf73f1a29"
ADAPTER_SHA256 = "6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b"


def _pinned_json(path: Path, expected_sha256: str):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Artifact identity mismatch")
    return json.loads(content)


def candidate_f_evaluation(serving_identity: dict) -> dict:
    empty = dict(available=False, split="development", n=None, turns=None,
                 candidate_status="selected_for_local_demo", assessment="ai_assessed_descriptive",
                 human_verified=False, full_success_rate=None, accepted_merge=False,
                 metrics=[], report_url=None)
    if (serving_identity.get("adapter_profile") != "quality-f"
            or serving_identity.get("adapter_model_sha256") != ADAPTER_SHA256):
        return {**empty, "quality_note": "Результат F недоступен: идентичность активного адаптера F не подтверждена."}
    try:
        data = _pinned_json(AGGREGATE, AGGREGATE_SHA256)
        verification = _pinned_json(VERIFICATION, VERIFICATION_SHA256)
        metrics = [dict(label=key, base=value["correct"],
                        fine_tuned=data["fine_tuned"]["turn_accuracy"][key]["correct"],
                        denominator=value["total"])
                   for key, value in data["base"]["turn_accuracy"].items()]
        outcomes = {mode: dict(successful=data[mode]["successful_cases"],
                               failed=data[mode]["failed_cases"], unknown=data[mode]["unknown_cases"])
                    for mode in ("base", "fine_tuned")}
        return {**empty, "available": True,
                "n": verification["independent_results"]["fine_tuned"]["planned_cases"],
                "turns": verification["independent_results"]["fine_tuned"]["planned_turns"],
                "method_status": data["method_status"], "metrics": metrics,
                "dialogue_outcomes": {"denominator": 100, **outcomes},
                "unresolved": {"criteria": data["null_criteria"], "rows": data["unresolved_rows"],
                               "paired_cases": verification["counts"]["adjudication_unresolved_pairs"]},
                "report_url": "/admin/candidate-f-report" if CANDIDATE_REPORT.is_file() else None,
                "quality_note": "F выбран пользователем для локального демо. 49 из 100 успешных диалогов — "
                "описательная оценка ИИ, не проверка человеком и не прохождение формального порога. "
                "Сохранены 4 неопределённых критерия в 4 ответах / 3 парах; эти диалоги уже имеют ошибки. "
                "QUALIFIED: был один операционный перезапуск; строгое соблюдение исходного retry-протокола "
                "не подтверждено. Результат не доказывает статистическое превосходство."}
    except (OSError, ValueError, KeyError, TypeError):
        return {**empty, "quality_note": "Результат F недоступен: сохранённый агрегат или итоговая проверка "
                "отсутствуют, повреждены либо не соответствуют закреплённой версии."}
