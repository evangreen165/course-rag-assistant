import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

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


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


query_engine = None


def build_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    Settings.embed_model = embed_model
    llm = build_llm()
    index, chroma_client = load_index(embed_model)
    retriever = build_hybrid_retriever(index, chroma_client, llm)
    qe = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
    qe.update_prompts({"response_synthesizer:text_qa_template": QA_TEMPLATE})
    return qe


app = FastAPI(title="RAG Assistant API")


@app.on_event("startup")
def startup_event():
    global query_engine
    query_engine = build_query_engine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    response = query_engine.query(request.question)
    answer = clean_response(response.response)
    sources = extract_sources(response.source_nodes)
    return AnswerResponse(answer=answer, sources=sources)
