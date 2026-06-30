"""
03_simple_rag_with_memory.py — Simple RAG with Conversation Memory
====================================================================

WHAT IS RAG WITH MEMORY?
    Stores previous conversations and uses them alongside new queries
    to generate better, context-aware answers. The system "remembers"
    what you talked about before.

    Without memory: "What about its limitations?" → "Whose limitations?"
    With memory:    "What about its limitations?" → Knows you were asking about transformers

HOW IT WORKS:
    1. Store conversation history (question + answer pairs)
    2. When a new question comes in, combine it with history
    3. Use the history to CONTEXTUALIZE the question before retrieval
    4. Retrieve documents based on the contextualized question
    5. Generate an answer using both retrieved docs AND conversation history

BEST FOR:
    Multi-turn chatbots, customer support, tutoring systems, any
    application where follow-up questions are expected.

KEY INSIGHT:
    The question "What about its limitations?" is meaningless without
    knowing what "it" refers to. Memory provides that context.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, delete_collection
from shared.llm import get_llm, ask_llm


COLLECTION_NAME = "rag_with_memory"


class ConversationMemory:
    """
    Simple conversation memory that stores chat history.

    In production, you'd use LangChain's ChatMessageHistory or a
    database-backed solution. We keep it simple here for clarity.

    TYPES OF MEMORY:
    - Buffer Memory (this): Stores all messages. Simple but grows unbounded.
    - Window Memory: Only keeps the last N exchanges. Bounded but may lose context.
    - Summary Memory: Summarizes old conversations. Best for long sessions.
    - Entity Memory: Tracks key entities mentioned. Good for complex topics.
    """

    def __init__(self, max_history: int = 10):
        self.history = []         # List of (HumanMessage, AIMessage) pairs
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str):
        """Store a Q&A exchange in memory."""
        self.history.append((
            HumanMessage(content=question),
            AIMessage(content=answer),
        ))
        # Trim to max history to prevent unbounded growth
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_messages(self):
        """Return all stored messages as a flat list."""
        messages = []
        for human_msg, ai_msg in self.history:
            messages.append(human_msg)
            messages.append(ai_msg)
        return messages

    def clear(self):
        """Clear all memory."""
        self.history = []

    def __len__(self):
        return len(self.history)


def format_docs(docs):
    """Join retrieved documents into a single string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def contextualize_question(question: str, memory: ConversationMemory) -> str:
    """
    Use the LLM to rewrite a follow-up question into a standalone question.

    This is the KEY TRICK of RAG with memory. Instead of searching for
    "What about its limitations?" (which would match nothing useful),
    we first rewrite it to "What are the limitations of transformers?"

    Args:
        question: The user's raw question (might be a follow-up).
        memory:   The conversation history.

    Returns:
        A standalone question that makes sense without context.
    """
    if len(memory) == 0:
        # First question — no context needed
        return question

    # Build the conversation history as text
    history_text = ""
    for human_msg, ai_msg in memory.history:
        history_text += f"User: {human_msg.content}\n"
        history_text += f"Assistant: {ai_msg.content[:200]}...\n\n"

    # Ask the LLM to rewrite the question
    prompt = f"""Given the following conversation history and a follow-up question,
rewrite the follow-up question to be a STANDALONE question that makes sense
without any conversation context.

Conversation History:
{history_text}

Follow-up Question: {question}

Standalone Question:"""

    standalone = ask_llm(
        prompt=prompt,
        system_prompt="You rewrite follow-up questions into standalone questions. Return ONLY the rewritten question, nothing else.",
    )

    return standalone.strip()


@timer
def build_rag_with_memory():
    """Build the RAG pipeline and return the components."""

    print_step(1, "Loading and chunking documents...")
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    print_result("Chunks created", str(len(chunks)))

    print_step(2, "Creating vector store...")
    delete_collection(COLLECTION_NAME)
    vector_store = create_vector_store(chunks, collection_name=COLLECTION_NAME)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print_result("Retriever ready", "top-4 semantic search")

    # ── The prompt includes a placeholder for conversation history ──
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI tutor. Answer questions based on the provided "
            "context. Be conversational and reference previous topics when relevant. "
            "If the context doesn't contain the answer, say so."
        )),
        # This placeholder will be filled with the conversation history
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )),
    ])

    llm = get_llm()
    memory = ConversationMemory()

    return retriever, prompt, llm, memory


def ask_with_memory(
    question: str,
    retriever,
    prompt,
    llm,
    memory: ConversationMemory,
) -> str:
    """
    Process a question using RAG with conversation memory.

    The flow is:
    1. Contextualize the question using history
    2. Retrieve documents using the contextualized question
    3. Generate answer using history + retrieved docs + question
    4. Store the exchange in memory
    """
    # ── Step 1: Contextualize ──
    contextualized_q = contextualize_question(question, memory)
    if contextualized_q != question:
        print(f"  🔄 Rewritten question: '{contextualized_q}'")

    # ── Step 2: Retrieve ──
    docs = retriever.invoke(contextualized_q)
    context = format_docs(docs)

    # ── Step 3: Generate ──
    messages = prompt.invoke({
        "chat_history": memory.get_messages(),
        "context": context,
        "question": question,  # Use ORIGINAL question for natural conversation
    })
    response = llm.invoke(messages)
    answer = response.content

    # ── Step 4: Store in memory ──
    memory.add_exchange(question, answer)

    return answer, docs


def run():
    """
    Demo: Multi-turn conversation showing how memory improves RAG.
    """
    print_header("🧠 RAG WITH MEMORY — Context-Aware Conversations")

    print("""
    This demo shows a multi-turn conversation where each question
    builds on previous ones. Watch how:
    ✓ Follow-up questions get rewritten to be standalone
    ✓ The AI references previous topics naturally
    ✓ "it", "they", "those" get resolved using conversation history
    """)

    # ── Build the system ──
    retriever, prompt, llm, memory = build_rag_with_memory()

    # ── Simulate a multi-turn conversation ──
    conversation = [
        "What are transformers in the context of AI?",
        "How does the self-attention mechanism work in them?",     # "them" = transformers
        "What about their limitations?",                           # "their" = transformers
        "How do embeddings relate to what we just discussed?",     # references the whole conversation
        "Which embedding models are best for RAG systems?",        # builds on embedding topic
    ]

    for i, question in enumerate(conversation, 1):
        print(f"\n{'─'*50}")
        print(f"  👤 Question {i}: {question}")
        print(f"{'─'*50}")
        print(f"  📝 Memory has {len(memory)} previous exchanges")

        answer, docs = ask_with_memory(question, retriever, prompt, llm, memory)

        print(f"\n  📄 Retrieved {len(docs)} documents")
        print_documents(docs, max_chars=100)

        print(f"  🤖 Answer:")
        print(f"  {answer}")

    # ── Summary ──
    print_header("📊 RAG WITH MEMORY — Key Takeaways")
    print("""
    What you saw:
    1. QUESTION REWRITING: "How does it work?" → "How does self-attention work in transformers?"
    2. CONTEXT AWARENESS: The AI knew "their" referred to transformers.
    3. CONVERSATION FLOW: Each answer built on previous ones naturally.

    Memory Types (from simple to complex):
    ┌─────────────────┬───────────────┬──────────────┬────────────┐
    │ Type            │ Stores        │ Scalability  │ Best For   │
    ├─────────────────┼───────────────┼──────────────┼────────────┤
    │ Buffer          │ All messages  │ Poor         │ Short chats│
    │ Window (K=5)    │ Last K msgs   │ Good         │ Most apps  │
    │ Summary         │ Summaries     │ Excellent    │ Long chats │
    │ Entity          │ Key entities  │ Good         │ Complex    │
    └─────────────────┴───────────────┴──────────────┴────────────┘

    Trade-offs:
    + More human-like, personalized interactions
    - Higher cost (more tokens per query due to history)
    - Privacy considerations (storing conversation data)
    """)


if __name__ == "__main__":
    run()
