# Запуск проекта через Docker

Docker запускает FastAPI, клиентский чат, админку, LangGraph и локальную Qwen с адаптером F в одном Linux-контейнере. Устанавливать Python и CUDA Toolkit на Windows для этого способа не нужно. NVIDIA-драйвер, Docker Desktop и WSL2 нужны на самом компьютере.

О назначении, сценариях и результатах: [что это за проект](PROJECT_OVERVIEW.md). [Что проверено на реальном ноутбуке](DOCKER_VERIFICATION.md).

## 1. Подготовить компьютер

Проверенная исходная машина: Windows, RTX 3080 Laptop с 16 ГБ видеопамяти. Docker использует GPU этой же машины; он не заменяет видеокарту и не ускоряет модель сам по себе. По умолчанию выбран BF16, при нехватке VRAM можно установить `INFERENCE_PRECISION=nf4` в `.env`.

- Установить/запустить [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) с Linux-контейнерами и WSL2 backend.
- Установить совместимый NVIDIA-драйвер Windows. Отдельный Linux display driver внутри WSL не ставить.
- Нужен интернет для первой сборки и скачивания весов. Ориентир свободного диска — не менее 35 ГБ: образ, около 8 ГБ Base, адаптер и временные файлы загрузки. При первичной загрузке скрипт временно хранит проверяемые файлы отдельно.
- Штатно остановить другие GPU-запуски проекта. Native Windows и Docker используют разные блокировки; контейнер не гарантирует обнаружение Windows-обучения или модели.

PowerShell:

```powershell
docker version
docker compose version
```

В `docker version` должны отображаться **Client и Server**. Если Server отсутствует, сначала устранить запуск Docker Desktop. [Официальные требования GPU/WSL2](https://docs.docker.com/desktop/features/gpu/).

## 2. Скачать код и собрать образ

```powershell
git clone https://github.com/Akana92/SaaS-ERP-CRM-Support-Client.git
cd SaaS-ERP-CRM-Support-Client
Copy-Item .env.example .env
docker compose build app
```

Сборка скачает официальный образ PyTorch/CUDA и установит зависимости. Это выполняется один раз или после изменения Dockerfile/зависимостей. В Docker image не копируются веса, токены, переписка и полный датасет.

Если checkout уже существует, перейти в него и выполнить `git pull --ff-only`. Существующий `.env` не перезаписывать без необходимости.

## 3. Получить веса

Источники закреплены в `configs/asset-sources.json`:

- Base: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Fine-tuned: [AkanaYB/saas-erp-support-qwen3-4b-adapter-f](https://huggingface.co/AkanaYB/saas-erp-support-qwen3-4b-adapter-f).
- TRAIN, отдельно от запуска чата: [AkanaYB/saas-erp-support-ru-train-v13](https://huggingface.co/datasets/AkanaYB/saas-erp-support-ru-train-v13).

Загрузчик использует точные commit SHA и проверяет размер/SHA-256 каждого файла. Он не заменяет уже существующие несовпадающие файлы. Наличие ссылок на HF-репозитории само по себе не означает, что туда уже загружены веса; если revision ещё `null`, публикацию соответствующего пакета нужно завершить.

Для публичных файлов достаточно пустого файла секрета:

```powershell
New-Item -ItemType Directory -Force .secrets | Out-Null
if (-not (Test-Path .secrets/hf_token)) {
    New-Item -ItemType File .secrets/hf_token | Out-Null
}
docker compose --profile setup run --rm assets
```

Это отдельный сервис загрузки без GPU. Он запишет Base в `models/qwen3_4b`, адаптер — в `adapters/quality-f`. Основному контейнеру эти папки доступны только для чтения. Проверенные файлы при следующем запуске повторно не скачиваются.

Если репозитории Private, записать read-токен в `.secrets/hf_token`. Файл исключён из Git и Docker build; секрет подключается только к сервису `assets`, не к работающему чату. Не указывать токен в `.env`, аргументах команды или README. [Права Hugging Face](https://huggingface.co/docs/hub/security-tokens).

### Если веса уже скачаны на этом ноутбуке

Можно указать существующие каталоги в `.env`, используя прямые слеши:

```dotenv
BASE_MODEL_DIR=D:/Agents/Projects/Capstone N4/models/qwen3_4b
ADAPTER_DIR=D:/Agents/Projects/Capstone N4/artifacts/stage8/ml-exports/model
INFERENCE_PRECISION=bf16
HF_TOKEN_FILE=./.secrets/hf_token
```

Затем проверить их загрузчиком без сети:

```powershell
docker compose --profile setup run --rm assets python scripts/fetch_assets.py --verify-only
```

Это проверяет исходные веса и подготовленный переносимый экспорт F. Пути других компьютеров будут отличаться. Не подставлять каталог checkpoints вместо `final_adapter`/проверенного экспорта.

## 4. Запустить приложение

Если до этого работал native-сервер проекта, остановить его из **исходной папки проекта**:

```powershell
.\.venv\Scripts\python.exe scripts/live_demo.py stop
.\.venv\Scripts\python.exe scripts/live_demo.py status
```

Дождаться `stopped` и освобождения порта/GPU. Затем в Docker-checkout:

```powershell
docker compose up -d app
docker compose logs -f app
```

Дождаться загрузки модели. `Ctrl+C` закрывает просмотр логов, контейнер продолжает работать.

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:7860/health
```

Готовность: контейнер `healthy`, в health `loaded=true`, `adapter_profile=quality-f`, `client_mode=fine_tuned`.

- **Клиент/сотрудник:** http://127.0.0.1:7860/client/
- **Администратор:** http://127.0.0.1:7860/admin/

Порт публикуется только на `127.0.0.1`; сервис доступен с этого компьютера. Это локальное демо без production-аутентификации. Не менять адрес публикации на `0.0.0.0` для доступа из интернета.

## 5. Что попробовать

1. В клиентском чате выбрать «Клиент»: «Прошла ли оплата по INV-1002?» — ответ должен опираться на фиктивный статус в ERP.
2. В новой беседе: «Оплатил INV-1001, четыре SIM не работают». Уточнения остаются в истории; при передаче в админке появляется карточка с причиной.
3. Выбрать «Сотрудник»: «Как запустить BP-2002?» — инструкция по демонстрационному процессу.
4. В админке открыть сравнение: отправить один вопрос обеим моделям, сопоставить текст, категорию, приоритет, настроение и действие. Base работает без адаптера, F — с адаптером. Сравнение выполняется последовательно на одной GPU.
5. Открыть trace и очередь: посмотреть этапы, токены, время и серверное решение. Это реальные локальные записи, не подтверждение успешного решения всех проблем.

Это ручные вымышленные примеры демо-ERP, а не закрытые вопросы итогового теста.

## 6. Остановить и перенести ноутбук

```powershell
docker compose exec app python scripts/live_demo.py stop
docker compose logs --tail 30 app
docker compose ps -a
```

Штатный stop ждёт текущую обработку и сохранение истории. Перед переносом убедиться, что контейнер завершился (`Exited`), а GPU освобождена. Можно также использовать `docker compose stop app`: Compose даёт до 10 минут на завершение, после чего может принудительно остановить процесс. Во время первоначальной загрузки весов ещё нет обычного HTTP-обработчика завершения; перед отправкой рабочих сообщений дождаться готовности.

Следующий запуск:

```powershell
docker compose up -d app
```

История/очередь находятся в named volume `saas-erp-support_live-data`, веса — в подключаемых папках. `docker compose down` удаляет контейнер, но сохраняет named volumes. **Не использовать `down -v` или очистку volumes**, если нужно сохранить переписку. Первая Docker-сессия имеет отдельную историю от native-сервера: старые пользовательские записи автоматически не импортируются.

## 7. Обновление проекта

Штатно остановить приложение, затем:

```powershell
git pull --ff-only
docker compose build app
docker compose up -d app
```

Веса и persistent volume остаются на месте. Изменение модели/схемы истории может потребовать отдельной миграции; не заменять сохранённые данные для обхода ошибки.

## Частые ошибки

| Симптом | Что проверить |
| --- | --- |
| Нет Docker Server / pipe `dockerDesktopLinuxEngine` | Docker Desktop запущен, выбран WSL2/Linux backend. Ошибка Docker Desktop до запуска контейнера не является ошибкой приложения. |
| GPU недоступна | NVIDIA-драйвер, WSL2, Docker GPU support. Выполнить `docker compose run --rm --no-deps app nvidia-smi`. |
| Port 7860 already allocated | Native-сервер или другой контейнер уже слушает порт; определить и штатно остановить его. |
| Bind source path does not exist | Сначала загрузить assets или исправить `.env`. App намеренно не создаёт пустые каталоги весов. |
| CUDA out of memory | Остановить другие GPU-задачи; попробовать NF4 после штатной остановки. Не загружать Base и F двумя отдельными серверами. |
| SHA-256 mismatch | Файл повреждён или относится к другой версии; сверить путь и manifest. Проверку SHA не отключать. |
| HF 401/403 | Проверить доступ к нужным репозиториям и read-токен. |
| CERTIFICATE_VERIFY_FAILED в Windows HF CLI | Использовать `python scripts/hf_windows.py auth login` в проектном venv: системное доверенное хранилище Windows, проверка HTTPS остаётся включённой. Не применять `verify=False`. |

На Linux-хосте вместо Docker Desktop требуется Docker Engine с настроенным NVIDIA Container Toolkit. Эта инструкция и local demo не являются готовой многопользовательской облачной поставкой. Финальное обучение требует полного training-окружения и отдельно разрешённой квитанции изоляции; Compose по умолчанию только обслуживает готовую модель.
