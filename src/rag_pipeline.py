"""RAG-пайплайн: ответы на вопросы студентов по локальному индексу ChromaDB.

Загружает индекс из коллекции ChromaDB (course_rag), эмбеддинги deepvk/USER2-base
и локальную LLM Qwen2.5-1.5B через llama.cpp (CPU). После каждого вопроса пишет
отчёт в папку reports/.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
REPORTS_DIR = PROJECT_ROOT / "reports"

COLLECTION_NAME = "course_rag"
EMBED_MODEL = "deepvk/USER2-base"
MODEL_FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
MODEL_PATH = PROJECT_ROOT / "modules" / MODEL_FILENAME

TOP_K = 3
CONTEXT_WINDOW = 4096
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

QA_TEMPLATE = PromptTemplate(
    "Ты — ассистент, который отвечает на вопросы студентов строго на основе "
    "предоставленного контекста.\n"
    "Отвечай ТОЛЬКО на основе предоставленного контекста. "
    "Если в контексте нет ответа, напиши: "
    "«В предоставленных материалах нет ответа на этот вопрос». "
    "Отвечай на русском языке.\n\n"
    "Контекст:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


def force_utf8_stdout():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    candidates = sorted((PROJECT_ROOT / "modules").glob("*.gguf"))
    if candidates:
        return candidates[0]
    return MODEL_PATH


def build_llm():
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    return LlamaCPP(
        model_path=str(model_path),
        context_window=CONTEXT_WINDOW,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        model_kwargs={"n_gpu_layers": 0, "n_threads": os.cpu_count() or 4},
        verbose=False,
    )


def load_index(embed_model):
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Папка ChromaDB не найдена: {CHROMA_DIR}. Сначала запустите src/ingest.py."
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Коллекция '{COLLECTION_NAME}' содержит {collection.count()} чанков.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def extract_sources(response):
    sources = []
    seen = set()
    for source_node in response.source_nodes:
        metadata = source_node.node.metadata or {}
        name = (
            metadata.get("file_name")
            or metadata.get("source")
            or metadata.get("file_path")
            or "неизвестный источник"
        )
        if name not in seen:
            seen.add(name)
            sources.append(name)
    return sources


def write_report(question, answer, sources):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"rag_report_{timestamp}.md"

    lines = [
        "# Отчёт RAG-пайплайна",
        "",
        f"- Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Модель LLM: {MODEL_FILENAME} (CPU, n_gpu_layers=0)",
        f"- Эмбеддинги: {EMBED_MODEL}",
        f"- Коллекция ChromaDB: {COLLECTION_NAME}",
        f"- similarity_top_k: {TOP_K}",
        f"- context_window: {CONTEXT_WINDOW}",
        "",
        "## Вопрос",
        "",
        question,
        "",
        "## Ответ",
        "",
        answer,
        "",
        "## Источники",
        "",
    ]
    if sources:
        for name in sources:
            lines.append(f"- {name}")
    else:
        lines.append("- (источники не найдены)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    force_utf8_stdout()

    print("Загружаю модель эмбеддингов...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    Settings.embed_model = embed_model

    print("Загружаю локальную LLM (CPU)...")
    llm = build_llm()

    print("Загружаю индекс из ChromaDB...")
    index = load_index(embed_model)

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=TOP_K)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_TEMPLATE})

    print("\nГотово. Введите вопрос (или 'выход' / 'exit' для завершения).")
    print("После каждого вопроса отчёт сохраняется в папку reports/.\n")

    while True:
        try:
            question = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение.")
            break

        if question.lower() in {"выход", "exit", "quit", "q"}:
            print("Завершение.")
            break
        if not question:
            continue

        response = query_engine.query(question)
        answer = str(response).strip()
        sources = extract_sources(response)

        print("\nОтвет:")
        print(answer)
        print("\nИсточники:")
        for name in sources:
            print(f"  - {name}")

        report_path = write_report(question, answer, sources)
        print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
