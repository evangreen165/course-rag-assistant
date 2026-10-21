import argparse
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
        input_dir=input_dir, recursive=True, filename_as_id=True
    ).load_data()
    print(f"Загружено файлов: {len(documents)}")
    return documents


def main():
    force_utf8_stdout()
    args = parse_args()

    documents = load_documents(args)
    if not documents:
        print("Документы не найдены. Завершение.")
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

    print("\n--- Первый чанк для проверки ---")
    print(nodes[0].text[:500])
    print("---")

    print(f"\nГотово: {len(nodes)} чанков сохранено в {CHROMA_DIR}")
    print("Повторный запуск скрипта очистит коллекцию и загрузит данные заново.")


if __name__ == "__main__":
    main()
