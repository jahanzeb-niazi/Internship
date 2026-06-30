"""
shared/llm.py — LLM Client Helpers
====================================
Wraps Google Gemini so every module can call `ask_llm(prompt)` without
repeating boilerplate.

KEY CONCEPT:
    An LLM (Large Language Model) is the "brain" of RAG. It takes a prompt
    (your question + retrieved context) and generates a human-like answer.
    We use Google Gemini here, but the pattern is the same for any LLM.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()


def get_llm(model: str = "gemini-2.0-flash", temperature: float = 0.0):
    """
    Create and return a Gemini LLM instance.

    Args:
        model: Which Gemini model to use.
                - "gemini-2.0-flash"  → Fast & cheap (good for learning)
                - "gemini-1.5-pro"    → More capable (for complex tasks)
        temperature: Controls randomness (0 = deterministic, 1 = creative).
                     We default to 0 for reproducible results.

    Returns:
        A LangChain ChatModel instance ready to use.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GOOGLE_API_KEY not found!\n"
            "   1. Go to https://aistudio.google.com/ and create an API key\n"
            "   2. Create a .env file in the project root\n"
            "   3. Add: GOOGLE_API_KEY=your-key-here"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def ask_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
) -> str:
    """
    Send a prompt to Gemini and get a text response.

    This is the simplest way to interact with the LLM. For RAG, we'll
    typically inject retrieved documents into the prompt before calling this.

    Args:
        prompt:        The user's question (possibly with context prepended).
        system_prompt: Instructions for how the LLM should behave.
        model:         Which Gemini model to use.
        temperature:   Randomness control.

    Returns:
        The LLM's text response as a string.

    Example:
        >>> answer = ask_llm("What is machine learning?")
        >>> print(answer)
        "Machine learning is a subset of AI..."
    """
    llm = get_llm(model=model, temperature=temperature)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


def ask_llm_with_context(
    question: str,
    context: str,
    system_prompt: str = (
        "You are a helpful assistant. Answer the question based ONLY on the "
        "provided context. If the context doesn't contain the answer, say "
        "'I don't have enough information to answer this question.'"
    ),
) -> str:
    """
    The core RAG function: answer a question using retrieved context.

    This is what makes RAG special — instead of relying on what the LLM
    "memorized" during training, we GIVE it the relevant information.

    Args:
        question: The user's question.
        context:  Retrieved documents/chunks joined as a string.
        system_prompt: Instructions constraining the LLM to use context.

    Returns:
        A grounded answer based on the provided context.
    """
    # ── Build the RAG prompt ──
    # We explicitly separate context from question so the LLM knows
    # which part is "source material" and which part is the query.
    prompt = f"""CONTEXT (retrieved documents):
{context}

QUESTION:
{question}

ANSWER:"""

    return ask_llm(prompt=prompt, system_prompt=system_prompt)
