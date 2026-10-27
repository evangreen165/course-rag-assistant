# Отчёт: REST API на FastAPI

- Дата: 2026-10-27
- Задача: REST API для RAG-ассистента.

## Созданные файлы

- `api/__init__.py` — пакет API.
- `api/main.py` — приложение FastAPI.

## Эндпоинты

| Метод | Путь | Описание |
| --- | --- | --- |
| POST | `/ask` | Принимает `QuestionRequest` (`question: str`), возвращает `AnswerResponse` (`answer: str`, `sources: list[str]`) |
| GET | `/health` | Возвращает `{"status": "ok"}` |

## Логика

- Приложение инициализируется как `FastAPI(title="RAG Assistant API")`.
- Модель (query_engine на Qwen 7B) загружается **один раз** при старте сервера через `@app.on_event("startup")` и хранится в глобальной переменной `query_engine`.
- В `/ask`: `query_engine.query(question)` → ответ очищается через `clean_response()` → уникальные имена файлов извлекаются из `response.source_nodes` через `extract_sources()`.
- Pydantic-модели: `QuestionRequest`, `AnswerResponse`.

## Запуск

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Проверка здоровья: `GET http://127.0.0.1:8000/health`
