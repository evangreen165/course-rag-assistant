"""Фабрика LLM: локальная модель (llama.cpp) или облачная (OpenAI-совместимая)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = PROJECT_ROOT / "modules"
LOCAL_MODEL_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

STOP_SEQUENCES = ["<|im_end|>", "\nВопрос:", "The original answer", "Original Answer", "The refined answer"]


def get_llm(provider, model_name=None, api_key=None, base_url=None):
    if provider == "local":
        from llama_index.llms.llama_cpp import LlamaCPP

        model_path = MODULES_DIR / LOCAL_MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Локальная модель не найдена: {model_path}")
        return LlamaCPP(
            model_path=str(model_path),
            context_window=8192,
            max_new_tokens=1024,
            temperature=0.1,
            generate_kwargs={"stop": STOP_SEQUENCES, "repeat_penalty": 1.2},
            model_kwargs={"n_gpu_layers": 0},
            completion_to_prompt=lambda text: text,
            verbose=False,
        )

    if provider == "cloud":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1024,
        )

    raise ValueError(f"Неизвестный провайдер: {provider}")
