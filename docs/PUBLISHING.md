# Публикация и получение проекта

Код: [Akana92/SaaS-ERP-CRM-Support-Client](https://github.com/Akana92/SaaS-ERP-CRM-Support-Client). Назначение и сценарии: [обзор проекта](PROJECT_OVERVIEW.md). Основной способ поставки окружения: [пошаговый Docker tutorial](DOCKER_TUTORIAL.md).

## Где что хранится

| Место | Содержимое |
| --- | --- |
| GitHub | Backend, frontend, обучение, конфигурации, CPU-тесты, агрегаты оценки, презентация, Dockerfile и Compose |
| [Hugging Face Model](https://huggingface.co/AkanaYB/saas-erp-support-qwen3-4b-adapter-f) | Публичный репозиторий адаптера F, tokenizer, карточки и SHA-манифеста |
| [Hugging Face Dataset](https://huggingface.co/datasets/AkanaYB/saas-erp-support-ru-train-v13) | Публичный репозиторий 13 финальных TRAIN-шардов v13, карточки и SHA-манифеста |
| Ноутбук | GPU, скачанные веса, история диалогов и очередь оператора |

**Оба ML-пакета опубликованы 19 сентября 2026 и доступны публично.** Model: commit `b0862d45d6a2bf71eb1578ac8ca91cdf6e5d5986` (12 файлов, 148 087 233 байта). Dataset: commit `84813aabfdaf1b9f85f17955128d5003872b7554` (15 файлов, 2 650 126 байт). Источник закреплённых ревизий и SHA-256 — [configs/asset-sources.json](../configs/asset-sources.json). Загрузчик использует точные commit SHA, а не ветку `main`. Границы проверки скачивания и запуска описаны в [DOCKER_VERIFICATION.md](DOCKER_VERIFICATION.md).

Base `Qwen/Qwen3-4B-Instruct-2507` скачивается из официального репозитория, ревизия `cdbee75f17c01a7cc42f958dc650907174af0554`. Базовые веса не дублируются в собственных репозиториях. GitHub и HF хранят файлы; публикация не запускает SaaS. Docker использует GPU компьютера, на котором его запустили. Docker image собран и проверен на RTX 3080 Laptop 16 ГБ: чат, админка, две связанные реплики и история после пересоздания контейнера.

## Повторная публикация пакетов владельцем

В исходном workspace подготовлены только разрешённые пакеты:

```text
artifacts/stage8/ml-exports/model/
artifacts/stage8/ml-exports/dataset/
```

Состав: [ML_ASSETS.md](ML_ASSETS.md). Не загружать целиком `artifacts`, `data` или `models`: в них есть закрытые и исторические материалы. Development/test, gold, raw-ответы оценки, переписка и checkpoints исключены.

Для upload нужен токен с правом записи в два указанных репозитория. Токен вводится только в интерактивном локальном входе; не передавать его в чат, аргументах команд или Git. В Windows использовать обёртку `scripts/hf_windows.py`: она подключает системное хранилище доверенных сертификатов через `truststore`, оставляя проверку HTTPS включённой. `truststore` включён в `requirements-stage2-lock.txt`.

Из **исходного workspace** с подготовленными пакетами:

```powershell
.\.venv\Scripts\python.exe scripts/hf_windows.py auth login
.\.venv\Scripts\python.exe scripts/hf_windows.py auth whoami
.\.venv\Scripts\python.exe scripts/hf_windows.py upload AkanaYB/saas-erp-support-qwen3-4b-adapter-f .\artifacts\stage8\ml-exports\model .
.\.venv\Scripts\python.exe scripts/hf_windows.py upload AkanaYB/saas-erp-support-ru-train-v13 .\artifacts\stage8\ml-exports\dataset . --repo-type dataset
```

После успешной загрузки сверить состав, сохранить полученные commit SHA в `configs/asset-sources.json` и проверить скачивание закреплённых файлов. Карточки и манифесты должны принадлежать той же версии, что веса/данные. [Права токенов](https://huggingface.co/docs/hub/security-tokens), [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Получение и запуск через Docker

[DOCKER_TUTORIAL.md](DOCKER_TUTORIAL.md) содержит подготовку Windows/WSL2, сборку, сервис `assets`, запуск, health-проверку, остановку и диагностику. Веса скачиваются отдельно и проверяются по SHA-256; Docker image не включает веса, TRAIN или токены. TRAIN для работы чата не нужен.

## Альтернативный запуск в Windows venv

```powershell
git clone https://github.com/Akana92/SaaS-ERP-CRM-Support-Client.git
cd SaaS-ERP-CRM-Support-Client
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-stage2-lock.txt
```

Нужен совместимый NVIDIA-драйвер. Исходный native-запуск проверялся на RTX 3080 Laptop 16 ГБ. Чистая установка на другом устройстве не проверена; CPU CI не подтверждает работу GPU.

Получить опубликованные Base и адаптер в пути native-приложения (для публичных пакетов вход Hugging Face не требуется):

```powershell
.\.venv\Scripts\python.exe scripts/fetch_assets.py --adapter-dir artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter
.\.venv\Scripts\python.exe scripts/fetch_assets.py --adapter-dir artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter --verify-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/live_demo.py start --adapter-profile quality-f --precision bf16 --attention sdpa_repeat_kv
Invoke-RestMethod http://127.0.0.1:7860/health
```

Дождаться `loaded=true`, затем открыть [чат](http://127.0.0.1:7860/client/) или [админку](http://127.0.0.1:7860/admin/). Загрузчик сверяет размеры и SHA-256 каждого разрешённого файла и не заменяет существующие несовпадающие файлы. Ожидаемый SHA256 весов адаптера: `6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b`.

```powershell
.\.venv\Scripts\python.exe scripts/live_demo.py stop
.\.venv\Scripts\python.exe scripts/live_demo.py status
```

Дождаться `stopped`. [Подробности native-запуска](LOCAL_RUN_AND_DELIVERY.md).

Для исследования TRAIN выполнить необязательную команду после настройки Windows venv:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_assets.py --adapter-dir artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter --dataset
```

Команда также проверяет Base и адаптер, скачивая недостающие файлы. Для TRAIN скрипт восстановит только разрешённые `data/quality90_v1/train/*.jsonl`, сохранив README проекта. SHA всех 13 файлов закреплены в конфигурации обучения и `configs/asset-sources.json`.

## Обучение, резервная копия и права

Адаптер и TRAIN — **не полный backup исследования**. Исторический запуск требует отдельной проверенной квитанции изоляции `data/quality90_v1/isolation_manifest_train_v13_attempt01.json` с SHA `1a918393bbc8b1727822d41182b844e3c89451939c7974a862217bc796fa697c`. Она не включена в публичный пакет; её проверка не отключается. Закрытая итоговая оценка требует независимой процедуры и защищённых данных. Не запускать старые builders поверх финальных данных v13.

Для продолжения прерванного обучения нужны checkpoints с optimizer/scheduler/RNG; они остаются локально. Резервную копию workspace делают отдельно. Эта поставка не запускает обучение или новую оценку.

Base лицензирована Apache-2.0. Лицензия собственного кода, адаптера и корпуса ещё не выбрана. Публичная доступность сама по себе не предоставляет коммерческих прав на все материалы; лицензия Base не назначается собственным материалам автоматически.
