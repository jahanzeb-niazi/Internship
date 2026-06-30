"""
05_query_rewriting.py — Query Rewriting
=========================================

WHAT IT IS:
    "Fix your question before asking it."
    Users write terrible search queries (typos, vague terms, missing context).
    Query rewriting uses an LLM to intercept the query and rewrite it into
    a highly optimized search string before sending it to the vector database.

STRATEGIES SHOWN HERE:
    1. Multi-Query: Turn 1 vague query into 3 specific variations.
    2. Step-Back Prompting: Turn a highly specific query into a broader conceptual one.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import print_header, print_step
from shared.llm import ask_llm

def demo_multi_query(original_query: str):
    print_step(1, "Multi-Query Generation")
    print(f"  Original (Vague): '{original_query}'")

    prompt = f"""You are an AI language model assistant. Your task is to generate 3 different
versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome
some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {original_query}"""

    response = ask_llm(prompt, "You generate search queries.").strip()
    queries = [q for q in response.split('\n') if q]

    print("\n  Rewritten Variations:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print("\n  (In production, you'd search all 3 and merge the results)")


def demo_step_back(original_query: str):
    print_step(2, "Step-Back Prompting")
    print(f"  Original (Too Specific): '{original_query}'")

    prompt = f"""You are an expert at extracting the core concept from a highly specific question.
Your task is to take the specific question and write a 'step-back' question. A step-back question
is a broader, more general question that encompasses the specific details.

Original question: {original_query}

Step-back question:"""

    step_back_query = ask_llm(prompt, "You abstract specific questions into broad concepts.").strip()

    print(f"\n  Step-Back Question:")
    print(f"  -> {step_back_query}")
    print("\n  (Searching for the broad concept often yields better background context)")

def run():
    print_header("✏️ QUERY REWRITING")

    # Often users ask things that don't match the terminology in the documents
    demo_multi_query("How do I fix the model making things up?")

    print("\n" + "="*50 + "\n")

    # Sometimes a query is so specific it misses the foundational principles
    demo_step_back("Why did the learning rate of 0.01 cause my ResNet50 on CIFAR-100 to diverge after epoch 5?")

if __name__ == "__main__":
    run()
