# Отчёт: RAGAS с облачным судьёй

- Дата: 2026-10-25
- Генератор: Qwen 2.5 7B (локально, llama.cpp, CPU)
- Судья: GLM-5.3-Flash (`z-ai/glm-5.3-flash`, OpenRouter)

## Метрики (по 5 тестовым вопросам)

| Вопрос | Answer Relevancy | Context Precision | Context Recall | Faithfulness |
| --- | ---: | ---: | ---: | ---: |
| Нормализация БД | 0.43 | 0.83 | — | — |
| Быстрая сортировка | 0.83 | 1.00 | — | — |
| git clean -n | 0.96 | 0.50 | — | — |
| Принципы ООП | 0.31 | 0.33 | — | — |
| TCP vs UDP | 0.60 | 0.64 | — | — |
| **Среднее** | **0.62** | **0.66** | **—** | **—** |

## Проблемы

1. **Faithfulness и Context Recall не посчитались** (NaN): судья GLM не завершал генерацию — `LLMDidNotFinishException` (не хватило `max_tokens` для сложных промптов), а также «LLM returned 1 generations instead of requested 3».
2. **Генератор (7B) местами выдаёт мусор**: китайские вводные, английские артефакты (`The refined answer`), повторы фраз.

## Вывод

Переход на облачного судью выполнен, оценка прогоняется за ~5 минут. Надёжно считаются **Answer Relevancy** и **Context Precision**; для **Faithfulness** и **Context Recall** нужно поднять `max_tokens` судьи или настроить формат вывода JSON.
