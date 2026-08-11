"""
LLM generation step.

Uses the OpenAI-compatible chat completions API. The API key and
model are read from environment variables so nothing is hardcoded;
swapping providers just means pointing OPENAI_BASE_URL at another
OpenAI-compatible endpoint (many providers support this).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI, APIError, APIConnectionError, AuthenticationError, RateLimitError

from rag.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from rag.retriever import RetrievedChunk


class GenerationError(Exception):
    """User-facing error for anything that goes wrong calling the LLM."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class GeneratedAnswer:
    text: str
    sources: List[str]
    retrieved_chunks: List[RetrievedChunk]


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")  # optional, for compatible providers

    if not api_key:
        raise GenerationError(
            "❌ No LLM API key configured. Set OPENAI_API_KEY in your environment (.env) "
            "before asking questions."
        )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def generate_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    conversation_history: Optional[List[dict]] = None,
    model: Optional[str] = None,
) -> GeneratedAnswer:
    if not retrieved_chunks:
        return GeneratedAnswer(
            text=(
                "I couldn't find enough relevant information in this repository "
                "to answer confidently."
            ),
            sources=[],
            retrieved_chunks=[],
        )

    client = _get_client()
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    context_block = build_context_block(retrieved_chunks)

    history_block = ""
    if conversation_history:
        recent = conversation_history[-6:]  # keep prompt bounded
        history_block = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent
        )

    user_prompt = build_user_prompt(question, context_block, history_block)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )
    except AuthenticationError:
        raise GenerationError("❌ LLM authentication failed. Check your OPENAI_API_KEY.")
    except RateLimitError:
        raise GenerationError("❌ LLM rate limit reached. Please wait a moment and try again.")
    except APIConnectionError:
        raise GenerationError("❌ Could not reach the LLM API. Please check your connection.")
    except APIError:
        raise GenerationError("❌ The LLM provider returned an error. Please try again.")
    except Exception:
        raise GenerationError("❌ Something went wrong generating the answer. Please try again.")

    answer_text = response.choices[0].message.content or ""

    seen = set()
    sources = []
    for chunk in retrieved_chunks:
        if chunk.file_path not in seen:
            seen.add(chunk.file_path)
            sources.append(chunk.file_path)

    return GeneratedAnswer(text=answer_text, sources=sources, retrieved_chunks=retrieved_chunks)
