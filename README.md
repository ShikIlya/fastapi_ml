# Churn Prediction Service

Сервис для обучения модели прогнозирования оттока клиентов (churn) и получения предсказаний через HTTP API. Сервис построен на FastAPI и scikit-learn.

## Возможности

- Проверка доступности сервиса и датасета через `GET /health`.
- Обучение модели через `POST /model/train`.
- Проверка состояния модели через `GET /model/status`.
- Получение метрик и истории обучения через `GET /model/metrics`.
- Получение схемы признаков через `GET /model/schema`.
- Прогноз оттока одного клиента через `POST /predict`.

Интерактивная документация Swagger UI доступна по адресу `/docs`, OpenAPI-схема — `/openapi.json`.

## Структура проекта

```text
api/                 HTTP-маршруты и описания ответов API
ml/pipeline.py       подготовка и обучение ML pipeline
schemas.py           Pydantic-схемы запросов и ответов
dataset.py           загрузка и подготовка датасета
model_storage.py     сохранение и загрузка обученной модели
training_history.py  хранение истории обучений
main.py              создание FastAPI-приложения
 data/churn_dataset.csv
models/              файлы обученной модели
```

## Датасет

Положите CSV-файл в `data/churn_dataset.csv`. Он должен содержать заголовок и следующие колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `monthly_fee` | число | Ежемесячная плата |
| `usage_hours` | число | Часы использования |
| `support_requests` | целое | Обращения в поддержку |
| `account_age_months` | целое | Возраст аккаунта в месяцах |
| `failed_payments` | целое | Неудачные платежи |
| `region` | строка | Регион |
| `device_type` | строка | Тип устройства |
| `payment_method` | строка | Способ оплаты |
| `autopay_enabled` | 0/1 | Включён ли автоплатёж |
| `churn` | 0/1 | Целевая метка: произошёл ли отток |

Пример заголовка:

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
```

`churn` используется для обучения и не передаётся в запрос `/predict`.

## Локальный запуск

Требуется Python 3.14 или совместимая версия, поддерживающая закреплённые версии зависимостей.

1. Создайте и активируйте виртуальное окружение.
2. Установите зависимости:

```bash
python -m pip install -r requirements.txt
```

3. Убедитесь, что файл `data/churn_dataset.csv` на месте.
4. Запустите API из корневой директории проекта:

```bash
python -m uvicorn main:app --reload
```

По умолчанию API будет доступно на `http://127.0.0.1:8000`.

## Запуск в Docker

Сборка образа из корневой директории проекта:

```bash
docker build -t churn-service .
```

Запуск:

```bash
docker run --rm -p 8000:8000 churn-service
```

Контейнер ожидает датасет в `data/churn_dataset.csv`. Чтобы использовать локальный датасет, смонтируйте каталог `data`:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" churn-service
```

На Windows PowerShell вместо `${PWD}` можно использовать `${PWD}` как путь текущей директории PowerShell; если Docker не принимает его, укажите полный путь к каталогу `data`.

Модель сохраняется в `models/churn_model.joblib`, история обучений — в `data/training_history.json`. Без постоянного тома эти изменения внутри контейнера пропадут после его удаления. Для сохранения смонтируйте каталоги данных и модели:

```bash
docker run --rm -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/models:/app/models" `
  churn-service
```

## API-примеры

### Обучить модель

`model_type` — `logreg` или `random_forest`. Допустимые гиперпараметры определяются моделью и проверяются схемой конфигурации обучения.

```bash
curl -X POST "http://127.0.0.1:8000/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "logreg",
    "hyperparameters": {}
  }'
```

Пример ответа:

```json
{
  "accuracy": 0.82,
  "f1": 0.68
}
```

Значения метрик выше приведены только как пример.

### Получить предсказание

Передайте все признаки кроме `churn`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 29.99,
    "usage_hours": 20.5,
    "support_requests": 1,
    "account_age_months": 12,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

Ответ содержит предсказанный класс и вероятности:

```json
{
  "churn": 0,
  "probability_stay": 0.84,
  "probability_churn": 0.16
}
```

### Другие маршруты

```text
GET /health
GET /model/status
GET /model/metrics?limit=5&model_type=logreg
GET /model/schema
```

## Тесты

Из корневой директории проекта:

```bash
python -m pytest tests -v
```
