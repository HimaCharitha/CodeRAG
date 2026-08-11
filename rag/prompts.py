"""Prompt templates for the generation step."""

SYSTEM_PROMPT = """You are an AI assistant analyzing a GitHub repository.

Answer questions using ONLY the retrieved repository context provided to you.

Do not invent files, functions, APIs, classes, dependencies, or implementation details.

If the retrieved context does not contain enough information to answer the question, clearly state that there is not enough information in the repository context.

When possible, mention the relevant file paths and explain how the retrieved code supports your answer.

Keep answers focused and technical. Use short code references (file paths, function/class names) rather than reproducing large blocks of code verbatim."""


def build_context_block(retrieved_chunks) -> str:
    """Render retrieved chunks into a single context string for the LLM."""
    if not retrieved_chunks:
        return "(no relevant context retrieved)"

    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(
            f"[{i}] File: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})\n"
            f"```{chunk.file_type}\n{chunk.content}\n```"
        )
    return "\n\n".join(parts)


def build_user_prompt(question: str, context_block: str, history_block: str = "") -> str:
    history_section = f"\nConversation so far:\n{history_block}\n" if history_block else ""
    return f"""Repository context retrieved for this question:

{context_block}
{history_section}
Question: {question}

Answer using only the context above. If it's insufficient, say so explicitly."""
