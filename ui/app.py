import json
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "ui" / "data"
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

CLOUD_MODELS = [
    "z-ai/glm-5.3-flash",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "openai/gpt-4o-mini",
]

DEFAULT_SETTINGS = {
    "provider": "local",
    "model_name": "z-ai/glm-5.3-flash",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "",
    "free_only": False,
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
    if provider == "local":
        return get_llm("local")
    return get_llm(
        "cloud",
        model_name=s.get("model_name"),
        api_key=s.get("api_key"),
        base_url=s.get("base_url"),
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
            persist_current_chat()
            st.session_state.current_chat_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()

        st.subheader("История чатов")
        for chat_id, chat in list(st.session_state.chat_history.items()):
            if st.button(chat.get("title", "Без названия"), key=f"chat_{chat_id}"):
                persist_current_chat()
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = list(chat.get("messages", []))
                st.rerun()

        with st.expander("⚙️ Настройки"):
            s = st.session_state.settings
            provider_labels = {"local": "Локальная", "cloud": "Облачная", "custom": "Кастомная"}
            provider = st.selectbox(
                "Провайдер",
                ["local", "cloud", "custom"],
                index=["local", "cloud", "custom"].index(s.get("provider", "local")),
                format_func=lambda p: provider_labels[p],
            )

            model_name = s.get("model_name", "")
            base_url = s.get("base_url", "")
            api_key = s.get("api_key", "")
            free_only = s.get("free_only", False)

            if provider == "cloud":
                free_only = st.checkbox("Только бесплатные модели", value=free_only)
                models = [m for m in CLOUD_MODELS if not free_only or ":free" in m]
                default_index = models.index(model_name) if model_name in models else 0
                model_name = st.selectbox("Модель", models, index=default_index)
                base_url = "https://openrouter.ai/api/v1"
                api_key = st.text_input("API-ключ (OpenRouter)", value=api_key, type="password")
            elif provider == "custom":
                base_url = st.text_input("Base URL", value=base_url)
                model_name = st.text_input("Model Name", value=model_name)
                api_key = st.text_input("API-ключ", value=api_key, type="password")

            if st.button("Сохранить настройки"):
                st.session_state.settings = {
                    "provider": provider,
                    "model_name": model_name,
                    "base_url": base_url,
                    "api_key": api_key,
                    "free_only": free_only,
                }
                save_json(SETTINGS_PATH, st.session_state.settings)
                st.session_state.query_engine = None
                st.success("Настройки сохранены")


def main():
    st.set_page_config(page_title="Ассистент по учебным материалам")
    st.title("📚 Ассистент по учебным материалам")

    init_state()
    render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Задайте вопрос по учебным материалам")

    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.query_engine is None:
            with st.spinner("Загружаю модель..."):
                llm = make_llm_from_settings(st.session_state.settings)
                st.session_state.query_engine = build_query_engine(llm)

        with st.chat_message("assistant"):
            with st.spinner("Генерирую ответ..."):
                response = st.session_state.query_engine.query(question)
                answer = clean_response(response.response)
                sources = extract_sources(response.source_nodes)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("Источники"):
            if sources:
                for name in sources:
                    st.markdown(f"- {name}")
            else:
                st.markdown("Источники не найдены.")

        persist_current_chat()


if __name__ == "__main__":
    main()
