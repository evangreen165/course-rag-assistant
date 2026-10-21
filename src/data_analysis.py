"""Анализ PDF-датасета в data/ и генерация паспорта docs/02_dataset_passport.md."""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "02_dataset_passport.md"


def force_utf8_stdout():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def collect_pdf_stats(data_dir: Path):
    stats = []
    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        rel = pdf_path.relative_to(data_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "(корень)"
        try:
            reader = PdfReader(str(pdf_path))
            pages = len(reader.pages)
            error = None
        except Exception as exc:
            pages = 0
            error = str(exc)
        stats.append(
            {
                "rel": rel,
                "category": category,
                "name": rel.name,
                "pages": pages,
                "error": error,
            }
        )
    return stats


def aggregate(stats):
    categories = defaultdict(
        lambda: {"files": 0, "pages": 0, "min": None, "max": None, "errors": 0}
    )
    for s in stats:
        c = categories[s["category"]]
        c["files"] += 1
        if s["error"]:
            c["errors"] += 1
            continue
        c["pages"] += s["pages"]
        c["min"] = s["pages"] if c["min"] is None else min(c["min"], s["pages"])
        c["max"] = s["pages"] if c["max"] is None else max(c["max"], s["pages"])
    return categories


def build_markdown(stats, categories):
    ok = [s for s in stats if not s["error"]]
    total_files = len(stats)
    total_pages = sum(s["pages"] for s in ok)
    error_files = [s for s in stats if s["error"]]

    lines = []
    lines.append("# Паспорт датасета")
    lines.append("")
    lines.append("| Параметр | Значение |")
    lines.append("| --- | --- |")
    lines.append(
        f"| Дата генерации | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC |"
    )
    lines.append("| Источник | `data/` |")
    lines.append(f"| Всего PDF-файлов | {total_files} |")
    lines.append(f"| Всего страниц (прочитано) | {total_pages} |")
    lines.append(f"| Категорий (папок верхнего уровня) | {len(categories)} |")
    lines.append(f"| Файлов с ошибками чтения | {len(error_files)} |")
    lines.append("")

    lines.append("## Статистика по категориям")
    lines.append("")
    lines.append(
        "| Категория | Файлов | Страниц | Мин. страниц | Макс. страниц | Средн. страниц |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cat in sorted(categories):
        c = categories[cat]
        readable = c["files"] - c["errors"]
        avg = round(c["pages"] / readable, 1) if readable else 0
        lines.append(
            f"| {cat} | {c['files']} | {c['pages']} | {c['min']} | {c['max']} | {avg} |"
        )
    lines.append("")

    lines.append("## Список файлов")
    lines.append("")
    lines.append("| Категория | Файл | Страниц |")
    lines.append("| --- | --- | ---: |")
    for s in stats:
        pages = s["pages"] if not s["error"] else f"ошибка: {s['error']}"
        lines.append(f"| {s['category']} | {s['name']} | {pages} |")
    lines.append("")

    lines.append("## Примечания")
    lines.append("")
    lines.append("- Паспорт сгенерирован автоматически скриптом `src/data_analysis.py`.")
    lines.append("- Учитываются только файлы `*.pdf`; архив `git.zip` и прочие форматы игнорируются.")
    lines.append("- Модель `modules/*.gguf` и прочие бинарные артефакты не входят в датасет.")
    lines.append("")
    return "\n".join(lines)


def main():
    force_utf8_stdout()
    if not DATA_DIR.is_dir():
        print(f"Ошибка: папка {DATA_DIR} не найдена")
        sys.exit(1)

    stats = collect_pdf_stats(DATA_DIR)
    if not stats:
        print("PDF-файлы не найдены.")
        sys.exit(1)

    categories = aggregate(stats)
    markdown = build_markdown(stats, categories)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Паспорт сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
