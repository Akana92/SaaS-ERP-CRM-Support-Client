# Публикация проекта: GitHub + Hugging Face

Подготовлено 19 сентября 2026 для `Akana92/SaaS-ERP-CRM-Support-Client`.

## Где что хранится

| Место | Содержимое | Для чего |
| --- | --- | --- |
| [GitHub](https://github.com/Akana92/SaaS-ERP-CRM-Support-Client) | Backend, frontend, код обучения, конфигурации, технические тесты, агрегаты оценки, презентация | Разработка, проверка кода, описание проекта |
| Hugging Face: Model, **Private** | Финальный адаптер F, tokenizer, карточка, SHA-манифест | Скачать и подключить Fine-tuned к Base |
| Hugging Face: Dataset, **Private** | Только 13 финальных TRAIN-шардов v13, карточка, SHA-манифест | Сохранить именно данные кандидата F |
| Ноутбук | Base, адаптер, окружение, GPU, история диалогов, локальная очередь | Запустить работающий чат и админку |

Это один проект с отдельным хранением крупных артефактов. Второй GitHub-репозиторий, MLflow, MinIO, Docker Registry и платный GPU-хостинг для этой поставки не требуются. Base `Qwen/Qwen3-4B-Instruct-2507` скачивается из официального репозитория; копию примерно 8 ГБ в собственный репозиторий не загружаем.

GitHub хранит код и может запускать CPU-тесты. Push не создаёт работающий SaaS: [GitHub Pages обслуживает статические файлы](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages), а здесь нужны FastAPI и GPU-модель. Model/Dataset в Hugging Face тоже являются хранилищами. Для нынешней демонстрации сервер остаётся на ноутбуке. Публичный постоянно работающий сайт с авторизацией — отдельный этап.

## Что сделать владельцу один раз

1. Зарегистрироваться или войти в [Hugging Face](https://huggingface.co/join). Имя пользователя может отличаться от GitHub.
2. Создать **Private Model** на [New model](https://huggingface.co/new): рекомендуемое имя `saas-erp-support-qwen3-4b-adapter-f`.
3. Создать **Private Dataset** на [New dataset](https://huggingface.co/new-dataset): рекомендуемое имя `saas-erp-support-ru-train-v13`.
4. В [Access Tokens](https://huggingface.co/settings/tokens) создать fine-grained токен с правом записи только в эти два репозитория. Не выбирать права на остальные проекты.
5. В локальном PowerShell из исходной папки проекта выполнить:

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

Вставить токен только в интерактивный запрос `login`. Добавлять его в Git credentials для CLI upload не требуется. Не отправлять токен в чат, не записывать в исходники и не указывать аргументом команды. Ассистенту передать **две ссылки на репозитории** и сообщить, что локальный вход выполнен. Тогда можно загрузить готовые пакеты и закрепить их commit SHA. [Официальная документация прав токена](https://huggingface.co/docs/hub/security-tokens).

На дату подготовки Free включает 100 ГБ private storage на аккаунт; остаток конкретного аккаунта не проверен. Подготовленные пакеты около 151 МБ суммарно укладываются в этот лимит при наличии свободной квоты. [Актуальные ограничения хранилища](https://huggingface.co/docs/hub/storage-limits).

## Готовые локальные пакеты и загрузка

В исходном workspace подготовлены:

```text
artifacts/stage8/ml-exports/model/
artifacts/stage8/ml-exports/dataset/
```

Состав и границы — [ML_ASSETS.md](ML_ASSETS.md). Эти папки не входят в Git. Закрытые development/test, gold, raw-ответы проверки, рабочая переписка, кэш и checkpoints в них не включены.

После создания Private-репозиториев и входа можно поручить загрузку ассистенту. Для самостоятельной загрузки из **исходного workspace**, заменив `YOUR_HF_LOGIN` своим именем:

```powershell
.\.venv\Scripts\hf.exe upload YOUR_HF_LOGIN/saas-erp-support-qwen3-4b-adapter-f .\artifacts\stage8\ml-exports\model .
.\.venv\Scripts\hf.exe upload YOUR_HF_LOGIN/saas-erp-support-ru-train-v13 .\artifacts\stage8\ml-exports\dataset . --repo-type dataset
```

Загружать именно подготовленные папки. Private следует установить при создании: `--private` не меняет видимость существующего репозитория. Карточки и манифесты должны попасть в ту же версию, что веса/данные. После загрузки сохранить commit SHA обоих HF-репозиториев; не ссылаться на изменяемый `main` как на воспроизводимую версию. [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli); команды upload/auth проверены по локальному `hf --help` версии 0.36.0.

## Как собрать всё на другом компьютере

### Код и окружение

```powershell
git clone https://github.com/Akana92/SaaS-ERP-CRM-Support-Client.git
cd SaaS-ERP-CRM-Support-Client
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-stage2-lock.txt
```

Нужен совместимый NVIDIA-драйвер. Проверенный ноутбук — RTX 3080 Laptop 16 ГБ. Чистая установка на другом устройстве ещё не проверена; CI проверяет CPU-контракты, а не загрузку GPU-модели.

### Base и адаптер

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/qwen3_4b
.\.venv\Scripts\hf.exe auth login
```

Затем заменить `YOUR_HF_LOGIN` и `MODEL_COMMIT_SHA` значениями после публикации:

```powershell
.\.venv\Scripts\hf.exe download YOUR_HF_LOGIN/saas-erp-support-qwen3-4b-adapter-f --revision MODEL_COMMIT_SHA --local-dir artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter
Get-FileHash artifacts/stage5/quality90-v1/training/candidate-f-v13-audited-labels/final_adapter/adapter_model.safetensors -Algorithm SHA256
```

Ожидаемый SHA256 весов адаптера: `6efc00de7bcecc24872b87de797abd4d9d43ea10e679b23d3b4c13ecb7f7109b`. Веса экспортированы без изменений; в копии adapter_config локальный путь заменён официальным ID и закреплённой revision Base. Для private-скачивания достаточно токена с чтением нужного репозитория.

### Запуск

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts/live_demo.py start --adapter-profile quality-f --precision bf16 --attention sdpa_repeat_kv
Invoke-RestMethod http://127.0.0.1:7860/health
```

Дождаться `loaded=true`, затем открыть клиент `http://127.0.0.1:7860/client/` и админку `http://127.0.0.1:7860/admin/`. Первая загрузка занимает время. Обе страницы обслуживает один сервер. [Подробное руководство и остановка](LOCAL_RUN_AND_DELIVERY.md).

### Датасет — только если нужен для исследования

Для работы чата TRAIN не нужен. Для получения его точной версии заменить имя и `DATASET_COMMIT_SHA`:

```powershell
.\.venv\Scripts\hf.exe download YOUR_HF_LOGIN/saas-erp-support-ru-train-v13 --repo-type dataset --revision DATASET_COMMIT_SHA --include "data/quality90_v1/train/*.jsonl" --local-dir .
```

Фильтр `--include` сохраняет нужные исходные пути и не перезаписывает README проекта карточкой датасета. SHA всех 13 файлов закреплены в `configs/train-quality90-f-v13-audited-labels.json`.

## Обучение, резервная копия и права

Адаптер и TRAIN дают рабочую модель и её обучающий корпус; это **не полный backup всего исследования**. Код обучения опубликован, но точный исторический запуск также требует отдельной проверенной квитанции изоляции `data/quality90_v1/isolation_manifest_train_v13_attempt01.json` с SHA `1a918393bbc8b1727822d41182b844e3c89451939c7974a862217bc796fa697c`. Она не включена в публичный пакет; её проверка не отключается. Закрытая итоговая оценка требует независимой процедуры и защищённых данных. Не запускать старые builders поверх финальных данных v13.

Для продолжения прерванного обучения нужны полные проверенные checkpoints с optimizer/scheduler/RNG, а не только `final_adapter`. Они остаются локально; резервную копию рабочего workspace следует делать отдельно на диск с достаточным свободным местом. Эта публикация не запускает новый цикл обучения или оценки.

Base лицензирована Apache-2.0. Лицензия собственного кода, адаптера и корпуса отдельно не выбрана: публичный GitHub не означает выдачу коммерческой лицензии на все материалы. Не присваиваем датасету права автоматически из лицензии Base. Поэтому ML-репозитории пока Private; выбор условий распространения — отдельное решение владельца.
