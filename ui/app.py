import glob
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "ui" / "data"
MODULES_DIR = PROJECT_ROOT / "modules"
CORPUS_DIR = PROJECT_ROOT / "data"
HISTORY_PATH = DATA_DIR / "history.json"
SETTINGS_PATH = DATA_DIR / "settings.json"

from llm_factory import get_llm  # noqa: E402
from rag_pipeline import (  # noqa: E402
    load_index,
    build_hybrid_retriever,
    clean_response,
    extract_sources,
    EMBED_MODEL,
    QA_TEMPLATE,
    SYSTEM_PROMPT,
)
from llama_index.core import Settings  # noqa: E402
from llama_index.core.llms import ChatMessage, MockLLM  # noqa: E402
from llama_index.core.prompts import ChatPromptTemplate  # noqa: E402
from llama_index.core.query_engine import RetrieverQueryEngine  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.llms.langchain import LangChainLLM  # noqa: E402

FALLBACK_CLOUD_MODELS = [
    "z-ai/glm-5.3-flash",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "openai/gpt-4o-mini",
]

DEFAULT_SETTINGS = {
    "provider": "local",
    "model_name": "",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "free_only": False,
    "max_new_tokens": 1024,
    "custom_providers": {},
}

CHAT_QA_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content="Контекст: {context_str}\nВопрос: {query_str}"),
    ]
)


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_data(ttl=600)
def fetch_openrouter_models(api_key=""):
    import requests

    headers = {}
    api_key = (api_key or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(
        "https://openrouter.ai/api/v1/models", timeout=10, headers=headers
    )
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []))


def list_local_models():
    return sorted(p.name for p in MODULES_DIR.glob("*.gguf"))


def count_tokens(text):
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


@st.cache_resource(show_spinner=False)
def get_retriever():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    Settings.embed_model = embed_model
    index, chroma_client = load_index(embed_model)
    return build_hybrid_retriever(index, chroma_client, MockLLM())


def to_llama_index_llm(llm):
    from llama_index.core.llms import LLM

    if isinstance(llm, LLM):
        return llm
    return LangChainLLM(llm=llm)


def build_query_engine(llm):
    llama_llm = to_llama_index_llm(llm)
    retriever = get_retriever()
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llama_llm)
    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": QA_TEMPLATE,
            "response_synthesizer:chat_content_qa_template": CHAT_QA_TEMPLATE,
        }
    )
    return query_engine


def make_llm_from_settings(s):
    provider = s.get("provider", "local")
    max_tokens = int(s.get("max_new_tokens", 1024))
    if provider == "local":
        return get_llm("local", model_name=s.get("model_name"), max_tokens=max_tokens)
    return get_llm(
        provider,
        model_name=s.get("model_name"),
        api_key=s.get("api_key"),
        base_url=s.get("base_url"),
        max_tokens=max_tokens,
    )


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_json(HISTORY_PATH, {})
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "settings" not in st.session_state:
        st.session_state.settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "stats" not in st.session_state:
        st.session_state.stats = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def persist_current_chat():
    if st.session_state.current_chat_id and st.session_state.messages:
        title = st.session_state.messages[0]["content"][:40] or "Новый чат"
        st.session_state.chat_history[st.session_state.current_chat_id] = {
            "title": title,
            "messages": st.session_state.messages,
        }
        save_json(HISTORY_PATH, st.session_state.chat_history)


def render_sidebar():
    with st.sidebar:
        if st.button("➕ Новый чат"):
            st.session_state.current_chat_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

        st.subheader("История чатов")
        for chat_id, chat_data in st.session_state.chat_history.items():
            if st.button(chat_data.get("title", "Без названия"), key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = list(chat_data.get("messages", []))
                st.session_state.pending_question = None
                st.rerun()

        with st.expander("⚙️ Настройки"):
            s = st.session_state.settings
            provider_labels = {"local": "Локальная", "cloud": "Облачная", "custom": "Кастомная"}
            provider = st.radio(
                "Провайдер",
                ["local", "cloud", "custom"],
                index=["local", "cloud", "custom"].index(s.get("provider", "local")),
                format_func=lambda p: provider_labels[p],
            )

            model_name = s.get("model_name", "")
            base_url = s.get("base_url", "")
            api_key = s.get("api_key", "")
            free_only = s.get("free_only", False)
            max_new_tokens = int(s.get("max_new_tokens", 1024))
            provider_name = ""
            custom_providers = s.get("custom_providers", {})

            if provider == "local":
                local_models = list_local_models()
                if not local_models:
                    st.warning("В папке modules/ нет .gguf-файлов.")
                else:
                    default_index = local_models.index(model_name) if model_name in local_models else 0
                    model_name = st.selectbox("Модель", local_models, index=default_index)
            elif provider == "cloud":
                free_only = st.checkbox("Только бесплатные модели", value=True)
                try:
                    all_models = fetch_openrouter_models(api_key)
                except Exception as e:
                    st.error(f"Ошибка загрузки моделей: {e}")
                    all_models = []
                models = [m for m in all_models if not free_only or ":free" in m]
                if not models:
                    st.warning("Не удалось загрузить список моделей. Проверьте интернет или API-ключ.")
                    model_name = st.text_input(
                        "Введите ID модели вручную", value=model_name or "z-ai/glm-5.3-flash"
                    )
                else:
                    default_index = models.index(model_name) if model_name in models else 0
                    model_name = st.selectbox("Модель", models, index=default_index)
                base_url = "https://openrouter.ai/api/v1"
                api_key = st.text_input("API-ключ (OpenRouter)", value=api_key, type="password")
            elif provider == "custom":
                provider_names = list(custom_providers.keys())
                if provider_names:
                    selected_provider = st.selectbox("Выбрать сохраненного провайдера", provider_names)
                    if selected_provider:
                        base_url = custom_providers[selected_provider].get("base_url", base_url)
                        model_name = custom_providers[selected_provider].get("model_name", model_name)
                        api_key = custom_providers[selected_provider].get("api_key", api_key)
                provider_name = st.text_input("Название провайдера (например, MyLocalAPI)", value="")
                base_url = st.text_input("Base URL", value=base_url)
                model_name = st.text_input("Model Name", value=model_name)
                api_key = st.text_input("API-ключ", value=api_key, type="password")

            max_new_tokens = st.slider("Макс. токенов (max_new_tokens)", 128, 2048, max_new_tokens)

            if st.button("Сохранить настройки"):
                if provider == "custom" and provider_name.strip():
                    if not (base_url.strip() and model_name.strip() and api_key.strip()):
                        st.error("Заполните Base URL, Model Name и API-ключ для кастомного провайдера.")
                    else:
                        custom_providers[provider_name.strip()] = {
                            "base_url": base_url.strip(),
                            "model_name": model_name.strip(),
                            "api_key": api_key.strip(),
                        }
                st.session_state.settings = {
                    "provider": provider,
                    "model_name": model_name.strip(),
                    "base_url": base_url.strip(),
                    "api_key": api_key.strip(),
                    "free_only": free_only,
                    "max_new_tokens": max_new_tokens,
                    "custom_providers": custom_providers,
                }
                save_json(SETTINGS_PATH, st.session_state.settings)
                st.session_state.query_engine = None
                st.success("Настройки сохранены")


def _ensure_chat(title):
    cid = st.session_state.current_chat_id
    if not cid:
        cid = uuid.uuid4().hex
        st.session_state.current_chat_id = cid
    if cid not in st.session_state.chat_history:
        st.session_state.chat_history[cid] = {"title": title, "messages": []}
    return cid


def _sync_chat():
    cid = st.session_state.current_chat_id
    if cid and cid in st.session_state.chat_history:
        st.session_state.chat_history[cid]["messages"] = st.session_state.messages
        save_json(HISTORY_PATH, st.session_state.chat_history)


def _answer(question):
    s = st.session_state.settings
    provider = s.get("provider", "local")

    with st.spinner("Готовлю модель..."):
        if "retriever" not in st.session_state:
            st.session_state.retriever = get_retriever()
        if provider != "local" or st.session_state.query_engine is None:
            llm = make_llm_from_settings(s)
            llama_llm = to_llama_index_llm(llm)
            Settings.llm = llama_llm
            query_engine = RetrieverQueryEngine.from_args(
                retriever=st.session_state.retriever, llm=llama_llm
            )
            query_engine.update_prompts(
                {
                    "response_synthesizer:text_qa_template": QA_TEMPLATE,
                    "response_synthesizer:chat_content_qa_template": CHAT_QA_TEMPLATE,
                }
            )
            st.session_state.query_engine = query_engine

    start = time.time()
    with st.chat_message("assistant"):
        with st.spinner("Генерирую ответ..."):
            response = st.session_state.query_engine.query(question)
            answer = clean_response(response.response)
            sources = extract_sources(response.source_nodes)
        st.markdown(answer)
    elapsed = time.time() - start

    model_label = s.get("model_name") or (
        "Qwen2.5-7B (локально)" if provider == "local" else "облачная модель"
    )
    st.session_state.stats.append(
        {
            "question": question,
            "answer": answer,
            "time": elapsed,
            "question_tokens": count_tokens(question),
            "answer_tokens": count_tokens(answer),
            "model": model_label,
            "context_size": len(sources),
        }
    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    _sync_chat()

    with st.expander("Источники"):
        if sources:
            for name in sources:
                st.markdown(f"- {name}")
        else:
            st.markdown("Источники не найдены.")


def render_chat_tab():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.get("pending_question")
    if pending:
        st.session_state.pending_question = None
        _answer(pending)
        return

    question = st.chat_input("Задайте вопрос по учебным материалам")

    if question and question.strip():
        question = question.strip()
        s = st.session_state.settings
        provider = s.get("provider", "local")
        if provider in ("cloud", "custom") and not (s.get("api_key") or "").strip():
            st.error("Ошибка: Введите API-ключ в настройках.")
            st.stop()

        title = question[:20] + ("..." if len(question) > 20 else "")
        _ensure_chat(title)
        st.session_state.messages.append({"role": "user", "content": question})
        _sync_chat()
        st.session_state.pending_question = question
        st.rerun()


def render_stats_tab():
    st.subheader("📊 Статистика")
    if not st.session_state.stats:
        st.info("Пока нет данных. Задайте вопрос в чате.")
        return

    last = st.session_state.stats[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Время ответа", f"{last['time']:.2f} с")
    col2.metric("Токены (вопрос/ответ)", f"{last['question_tokens']}/{last['answer_tokens']}")
    col3.metric("Модель", last["model"])
    col4.metric("Размер контекста", f"{last['context_size']} чанков")

    st.divider()
    st.markdown("### История запросов")
    for i, entry in enumerate(reversed(st.session_state.stats)):
        st.markdown(
            f"**{i + 1}.** `{entry['model']}` — {entry['time']:.2f} с, "
            f"токены {entry['question_tokens']}/{entry['answer_tokens']} — "
            f"_{entry['question'][:60]}_"
        )


def render_materials_tab():
    st.subheader("📚 Материалы")

    if st.button("Переиндексировать корпус"):
        with st.spinner("Индексация... (на CPU это может занять часы)"):
            result = subprocess.run(
                [sys.executable, "src/ingest.py"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
        if result.returncode == 0:
            st.success("Индексация завершена.")
        else:
            st.error("Ошибка индексации.")
            st.code((result.stdout or "") + (result.stderr or ""))

    uploaded = st.file_uploader("Загрузить новые PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            dest = CORPUS_DIR / f.name
            dest.write_bytes(f.getbuffer())
        st.info("Файлы сохранены в data/. Нажмите «Переиндексировать корпус».")

    st.subheader("Текущие файлы")
    files = sorted(CORPUS_DIR.rglob("*.pdf"))
    if not files:
        st.info("В data/ нет PDF-файлов.")
    for p in files:
        size_kb = p.stat().st_size / 1024
        st.markdown(f"- `{p.relative_to(CORPUS_DIR)}` ({size_kb:.1f} КБ)")


def main():
    st.set_page_config(page_title="Ассистент по учебным материалам")
    st.title("📚 Ассистент по учебным материалам")

    init_state()
    render_sidebar()

    tab_chat, tab_stats, tab_materials = st.tabs(["💬 Чат", "📊 Статистика", "📚 Материалы"])
    with tab_chat:
        render_chat_tab()
    with tab_stats:
        render_stats_tab()
    with tab_materials:
        render_materials_tab()


if __name__ == "__main__":
    main()
