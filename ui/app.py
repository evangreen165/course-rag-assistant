import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import (  # noqa: E402
    build_llm,
    load_index,
    build_hybrid_retriever,
    clean_response,
    extract_sources,
    EMBED_MODEL,
    QA_TEMPLATE,
)
from llama_index.core import Settings  # noqa: E402
from llama_index.core.query_engine import RetrieverQueryEngine  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402


@st.cache_resource(show_spinner=False)
def build_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    Settings.embed_model = embed_model
    llm = build_llm()
    index, chroma_client = load_index(embed_model)
    retriever = build_hybrid_retriever(index, chroma_client, llm)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_TEMPLATE})
    return query_engine


st.set_page_config(page_title="Ассистент по учебным материалам")
st.title("📚 Ассистент по учебным материалам")

with st.spinner("Загружаю RAG-систему (модель и индекс)... Это может занять минуту."):
    query_engine = build_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Задайте вопрос по учебным материалам")

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ..."):
            response = query_engine.query(question)
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
