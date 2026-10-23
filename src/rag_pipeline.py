"""RAG-пайплайн с гибридным поиском (векторный + BM25).

Загружает индекс из коллекции ChromaDB (course_rag), эмбеддинги deepvk/USER2-base
и локальную LLM Qwen2.5-1.5B через llama.cpp (CPU). Для поиска использует
QueryFusionRetriever (векторный ретривер + BM25). После каждого вопроса пишет
отчёт в папку reports/.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
REPORTS_DIR = PROJECT_ROOT / "reports"

COLLECTION_NAME = "course_rag"
EMBED_MODEL = "deepvk/USER2-base"
MODEL_FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
MODEL_PATH = PROJECT_ROOT / "modules" / MODEL_FILENAME

TOP_K = 3
FUSION_TOP_K = 4
CONTEXT_WINDOW = 4096
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.1

STOP_SEQUENCES = ["<|im_end|>", "\nВопрос:", "The original answer", "The refined answer", "Original Answer:"]

SYSTEM_PROMPT = (
    "Ты — ассистент по учебным материалам. Отвечай на вопрос студента, используя "
    "ТОЛЬКО предоставленный контекст.\n"
    "Не пиши никаких вводных слов, не выводи JSON, не копируй метаданные. "
    'Пиши только чистый текст ответа. Если ответа нет, напиши "В предоставленных '
    'материалах нет ответа на этот вопрос". '
    "Отвечай ТОЛЬКО на русском языке. Не пиши никаких комментариев. Просто дай ответ."
)

QA_TEMPLATE = PromptTemplate(
    "<|im_start|>system\n"
    + SYSTEM_PROMPT
    + "<|im_end|>\n"
    "<|im_start|>user\n"
    "Контекст: {context_str}\nВопрос: {query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def force_utf8_stdout():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def identity_prompt(text: str) -> str:
    return text


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
        generate_kwargs={"stop": STOP_SEQUENCES, "repeat_penalty": 1.3},
        model_kwargs={"n_gpu_layers": 0, "n_threads": os.cpu_count() or 4},
        completion_to_prompt=identity_prompt,
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
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index, chroma_client


def get_all_nodes(chroma_client):
    """Восстанавливает все ноды из ChromaDB постранично (docstore при загрузке пуст)."""
    collection = chroma_client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Всего документов в коллекции: {total}")

    nodes = []
    batch_size = 5000
    for offset in range(0, total, batch_size):
        data = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for text, meta in zip(data["documents"], data["metadatas"]):
            nodes.append(TextNode(text=text, metadata=meta or {}))
        print(f"Загружено нод: {len(nodes)}/{total}")

    return nodes


def build_hybrid_retriever(index, chroma_client, llm):
    nodes = get_all_nodes(chroma_client)
    print(f"BM25 построен на {len(nodes)} нодах.")

    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    vector_retriever = index.as_retriever(similarity_top_k=TOP_K)

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        llm=llm,
        similarity_top_k=FUSION_TOP_K,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
        verbose=True,
    )


def extract_sources(nodes):
    sources = []
    seen = set()
    for node in nodes:
        metadata = node.node.metadata or {}
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
        f"- Гибридный поиск: vector + BM25 (RRF, top_k={FUSION_TOP_K})",
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
    index, chroma_client = load_index(embed_model)

    print("Строю гибридный ретривер (vector + BM25)...")
    retriever = build_hybrid_retriever(index, chroma_client, llm)

    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
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
        sources = extract_sources(response.source_nodes)

        print("\n--- Найденные фрагменты ---")
        for i, node in enumerate(response.source_nodes, start=1):
            text = (node.node.get_content() or "").strip()
            print(f"[{i}] {text[:200]}")
        print("---")

        print("\nОтвет:")
        print(answer)
        print("\nИсточники:")
        for name in sources:
            print(f"  - {name}")

        report_path = write_report(question, answer, sources)
        print(f"\nОтчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
