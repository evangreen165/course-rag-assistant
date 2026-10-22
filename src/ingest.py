import argparse
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "course_rag"
EMBED_MODEL = "deepvk/USER2-base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def force_utf8_stdout():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Загрузка учебных материалов в ChromaDB")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Подпапка внутри data/ (например, git). Если не указана — читается весь data/",
    )
    return parser.parse_args()


def is_garbage(text: str) -> bool:
    """Определяет, является ли извлечённый текст бинарным мусором."""
    if not text or not text.strip():
        return True
    if "%PDF" in text or "\x00" in text:
        return True
    printable = sum(1 for ch in text if ch.isprintable() or ch.isspace())
    return printable / max(len(text), 1) < 0.8


def filter_garbage(documents):
    """Разделяет документы на корректные и мусорные."""
    good = []
    discarded = []
    for doc in documents:
        text = doc.get_content() or doc.text or ""
        if is_garbage(text):
            discarded.append(doc)
        else:
            good.append(doc)
    return good, discarded


def load_documents(args):
    if args.folder:
        input_dir = DATA_DIR / args.folder
        if not input_dir.is_dir():
            print(f"Ошибка: папка {input_dir} не существует")
            sys.exit(1)
        label = f"data/{args.folder}/"
    else:
        input_dir = DATA_DIR
        label = "data/ (все подпапки)"

    print(f"Читаю документы из: {label}")
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        filename_as_id=True,
        file_extractor={".pdf": PyMuPDFReader()},
    ).load_data()
    print(f"Загружено документов (страниц): {len(documents)}")
    return documents


def main():
    force_utf8_stdout()
    args = parse_args()

    documents = load_documents(args)
    if not documents:
        print("Документы не найдены. Завершение.")
        sys.exit(0)

    documents, discarded = filter_garbage(documents)
    all_files = {d.metadata.get("file_name") for d in documents} | {
        d.metadata.get("file_name") for d in discarded
    }
    good_files = {d.metadata.get("file_name") for d in documents}
    discarded_files = all_files - good_files
    print(f"Всего файлов: {len(all_files)}")
    print(f"Успешно прочитано файлов: {len(good_files)}")
    print(f"Отброшено файлов: {len(discarded_files)}")
    print(f"Отброшено мусорных документов (страниц): {len(discarded)}")
    if not documents:
        print("После фильтрации не осталось корректных документов. Завершение.")
        sys.exit(0)

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Получено чанков: {len(nodes)}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Старая коллекция удалена (очистка перед загрузкой)")
    except Exception:
        print("Коллекция ещё не существует — создаём новую")

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )

    print("\n--- Первый чанк для проверки (первые 200 символов) ---")
    print(nodes[0].text[:200])
    print("---")

    print(f"\nГотово: {len(nodes)} чанков сохранено в {CHROMA_DIR}")
    print("Повторный запуск скрипта очистит коллекцию и загрузит данные заново.")


if __name__ == "__main__":
    main()
