"""Оценка качества RAG-пайплайна метриками RAGAS.

Генератор — Qwen 7B (llama-index LlamaCPP), судья — GLM-5.3-Flash (OpenRouter).
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import (  # noqa: E402
    build_llm,
    load_index,
    build_hybrid_retriever,
    EMBED_MODEL,
    QA_TEMPLATE,
)
from llama_index.core import Settings  # noqa: E402
from llama_index.core.query_engine import RetrieverQueryEngine  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "test_questions.jsonl"
RESULTS_CSV = PROJECT_ROOT / "reports" / "ragas_results.csv"


def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def build_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, embed_batch_size=32)
    Settings.embed_model = embed_model
    generator_llm = build_llm()
    index, chroma_client = load_index(embed_model)
    retriever = build_hybrid_retriever(index, chroma_client, generator_llm)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=generator_llm)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_TEMPLATE})
    return query_engine


def build_judge_llm():
    return ChatOpenAI(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0,
        max_tokens=1024,
    )


def main():
    load_dotenv()
    questions = load_questions(QUESTIONS_PATH)
    if not questions:
        raise SystemExit(f"Нет вопросов в {QUESTIONS_PATH}")

    query_engine = build_query_engine()

    records = []
    for item in questions:
        question = item["question"]
        ground_truth = item.get("ground_truth") or item.get("answer")
        response = query_engine.query(question)
        answer = response.response.strip()
        contexts = [n.node.get_content() for n in response.source_nodes]
        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
        print(f"Q: {question}\nA: {answer}\n---")

    judge_llm = LangchainLLMWrapper(build_judge_llm())
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=1800),
    )

    df = result.to_pandas()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"Результаты сохранены в {RESULTS_CSV}")
    print(df)


if __name__ == "__main__":
    main()
