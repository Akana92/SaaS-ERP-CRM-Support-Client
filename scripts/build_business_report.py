"""Build a shareable, text-free measurement report from the authored live smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support.business_metrics import summarize_smoke


def render(report):
    c, usage = report["client"], report["client"]["usage"]
    latency = usage["latency"]
    number = lambda value: "не измерено" if value is None else f"{value:,.2f}".replace(",", " ")
    seconds = lambda value: number(value / 1000 if value is not None else None)
    lines = ["# Бизнес-метрики: локальный демонстрационный прогон", "",
             "Это описательная статистика авторских ERP-сценариев, а не оценка на клиентах или итоговом test.", "",
             f"Выборка: **{c['requests']} уникальных сообщений / {c['conversations']} диалога**. "
             "Уточнение — отдельное сообщение внутри той же беседы. Повторная отправка с тем же request_id не учитывается второй раз.", "",
             "| Наблюдение | Значение |", "| --- | ---: |",
             f"| Полнота учёта расхода | {'полная' if usage['complete'] else 'неполная'} |",
             f"| Сообщения без записи расхода | {usage['requests_without_usage']} |",
             f"| Вызовы модели в клиентском чате | {usage['calls']} |",
             f"| Входные токены | {number(usage['input_tokens'])} |",
             f"| Выходные токены | {number(usage['output_tokens'])} |",
             f"| Всего токенов | {number(usage['total_tokens'])} |",
             f"| Среднее на вызов | {number(usage['tokens_per_call'])} |",
             f"| Фактическая оплата API | {number(usage['api_cost'])} USD |",
             f"| Среднее время генерации | {seconds(latency['mean_ms'])} с |",
             f"| Медиана генерации | {seconds(latency['median_ms'])} с |",
             f"| Минимум / максимум генерации | {seconds(latency['min_ms'])} / {seconds(latency['max_ms'])} с |",
             f"| Ответ без передачи оператору | {c['answered_without_handoff']}/{c['requests']} "
             f"({c['answered_without_handoff_rate']:.0%}) |",
             f"| Сообщения с передачей оператору | {c['escalated_requests']}/{c['requests']} "
             f"({c['escalated_request_rate']:.0%}) |",
             f"| Диалоги хотя бы с одной передачей | {c['conversations_with_handoff']}/{c['conversations']} "
             f"({c['conversation_handoff_rate']:.0%}) |", "",
             "Отсутствие передачи не означает подтверждённого решения. Повторная передача в одной беседе "
             "не создаёт второй диалог в знаменателе. Время — внутри generate: очередь, подготовка и доставка ответа не включены. "
             "Первый токен и p95 на этой малой выборке не заявляются.", "",
             f"Raw-решение модели о передаче: да {c['raw_model_handoff']['yes']}, нет {c['raw_model_handoff']['no']}, "
             f"нет валидного решения {c['raw_model_handoff']['unavailable']}. "
             "Маршрут сервера считается отдельно: он учитывает бизнес-правила и резервную передачу при ошибке модели.", "",
             "## Сравнение моделей — отдельный диагностический расход", ""]
    pair = report["comparison"]
    if pair["available"]:
        lines.extend(["| Режим | Вход | Выход | Всего | Генерация, с | API, USD |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for mode in ("base", "fine_tuned"):
            u = pair[mode]
            lines.append(f"| {mode} | {number(u['input_tokens'])} | {number(u['output_tokens'])} | "
                         f"{number(u['total_tokens'])} | {seconds(u['latency']['mean_ms'])} | {number(u['api_cost'])} |")
        lines.extend(["", "Эти два вызова исключены из клиентских знаменателей. Пара не является повторной оценкой качества F."])
    else:
        lines.append("Завершённая пара отсутствует; клиентские метрики не подменяются частичными результатами сравнения.")
    lines.extend(["", "## Что пока не измерено", "",
                  "Подтверждённое решение проблемы, экономия времени оператора, человеческая исходная скорость, "
                  "электроэнергия и полная себестоимость — **не измерены**. 0 USD означает отсутствие оплаты API; "
                  "время GPU и электричество не объявляются бесплатными.", "",
                  "## Источник и воспроизведение", "",
                  f"Источник: `{report['source']['path']}`; SHA-256 `{report['source']['sha256']}`.",
                  "Исходный JSON содержит только новые вымышленные сценарии ручной проверки. В этот отчёт не переносились "
                  "сообщения, ответы, персональные данные, идентификаторы бесед или локальные пути весов.",
                  "`python scripts/build_business_report.py` повторяет расчёт на исходном workspace; "
                  "в лёгком пакете исходный журнал не распространяется, предоставлены агрегаты и unit-тесты формул.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/stage5/latency-2026-09-19/http/http-smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    source, output = (ROOT / args.source).resolve(), (ROOT / args.output).resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        parser.error("Paths must remain inside the project")
    content = source.read_bytes()
    report = summarize_smoke(json.loads(content))
    report["source"] = {"path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(content).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "business-metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "business-metrics.md").write_text(render(report), encoding="utf-8")
    template = output / "operator-time-study-template.csv"
    # Never erase filled human observations on report rebuild.
    if not template.exists():
        with template.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerow(["case_id", "operator_pseudonym", "mode", "task_family", "active_seconds",
                                        "elapsed_seconds", "resolved", "correctness_reviewed", "notes"])
    print(json.dumps({"requests": report["client"]["requests"], "conversations": report["client"]["conversations"],
                      "tokens": report["client"]["usage"]["total_tokens"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
