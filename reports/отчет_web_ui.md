# Отчёт: веб-интерфейс Streamlit

- Дата: 2026-10-25
- Задача: веб-интерфейс для RAG-ассистента + исправление оценки RAGAS.

## Изменения

1. **Оценщик RAGAS (`eval/ragas_eval.py`)**:
   - `max_tokens` судьи (GLM) увеличен с 1024 до **4096** — чтобы успевал анализировать длинные контексты и корректно закрывать JSON.
   - `AnswerRelevancy(strictness=1)` — решает ошибку «LLM returned 1 generations instead of requested 3».

2. **Генератор (`src/rag_pipeline.py`)**:
   - Добавлена функция `clean_response()` — отрезает английские/китайские артефакты (`The original answer`, `The refined answer`, `Original Answer:`, `回答`) перед возвратом ответа.
   - Ответ теперь формируется как `clean_response(response.response)`.

3. **Веб-интерфейс (`ui/app.py`)**:
   - Streamlit-приложение с заголовком «📚 Ассистент по учебным материалам».
   - Чат через `st.chat_message` + `st.chat_input`, история в `st.session_state`.
   - Ответ выводится в чате, источники (`source_nodes`) — в `st.expander`.
   - RAG-система загружается один раз через `@st.cache_resource`.
   - Пустой ввод обрабатывается.

## Запуск

```powershell
.venv\Scripts\python.exe -m streamlit run ui\app.py
```
