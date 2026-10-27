# Отчёт: полноценный чат-интерфейс

- Дата: 2026-10-27
- Задача: превратить Streamlit-приложение в продукт уровня ChatGPT.

## Что сделано

1. **`src/llm_factory.py`** — фабрика `get_llm(provider, model_name, api_key, base_url)`:
   - `local` → llama-index `LlamaCPP` (Qwen2.5-7B, `n_gpu_layers=0`, `max_new_tokens=1024`);
   - `cloud` → `ChatOpenAI` (langchain_openai) с переданными `base_url`/`api_key`.

2. **`ui/app.py`** — полный рерайт:
   - боковая панель: «➕ Новый чат», список прошлых чатов, раздел «⚙️ Настройки»;
   - выбор провайдера: Локальная / Облачная / Кастомная; для облачной — выпадающий список моделей (OpenRouter/OpenAI/Gemini) и чекбокс «Только бесплатные модели» (фильтр по `:free`);
   - `st.session_state`: `messages`, `chat_history`, `current_chat_id`, `settings`, `query_engine`;
   - сохранение/загрузка в `ui/data/history.json` и `ui/data/settings.json`;
   - основной чат: `st.chat_input` → `query_engine.query()` → ответ + источники в `st.expander`;
   - смена модели через `Settings.llm` + пересборка `query_engine` (облачная модель оборачивается `LangChainLLM`).

3. **Зависимость** `llama-index-llms-langchain` добавлена в `requirements.txt`.

4. `ui/data/` добавлена в `.gitignore` (там API-ключи и история — не коммитим).

## Запуск

```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```
