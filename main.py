#!/usr/bin/env python3
"""
Learning Agent CLI
--------------------
Gemini (default):  python main.py "weather in Tokyo"
Mock (offline):    python main.py --mock "list my tasks"
Interactive:       python main.py
"""

from __future__ import annotations

import argparse
import sys

from config import get_gemini_api_key
from gemini_llm import GeminiLLM
from mock_llm import MockLLM
from react_agent import ReActAgent
from tools import pretty_tool_list


DEMO_QUERIES = [
    "What's the weather in Tokyo?",
    "List my tasks",
    "Add task: finish Day 4 homework",
    "Delete task 2",
]


def create_agent(use_mock: bool) -> ReActAgent:
    if use_mock:
        return ReActAgent(llm=MockLLM(), verbose=True, backend_name="mock")

    if not get_gemini_api_key():
        print("ERROR: GEMINI_API_KEY not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Paste your key from https://aistudio.google.com/apikey")
        print("  3. Or run with --mock for offline mode")
        sys.exit(1)

    return ReActAgent(llm=GeminiLLM(), verbose=True, backend_name="gemini")


def interactive_mode(agent: ReActAgent) -> None:
    print(pretty_tool_list())
    print("\nType a question, 'demo' to run examples, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "demo":
            run_demo(agent)
            continue

        agent.run(user_input)
        print()


def run_demo(agent: ReActAgent) -> None:
    print("\n>>> Running demo queries (approve/deny when prompted)\n")
    for q in DEMO_QUERIES:
        print(f"\n{'#' * 60}\nDEMO QUERY: {q}\n{'#' * 60}")
        agent.run(q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning agent with ReAct + tool calling")
    parser.add_argument("query", nargs="*", help="One-shot question")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use keyword mock LLM instead of Gemini (no API key)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_agent(use_mock=args.mock)

    if args.query:
        agent.run(" ".join(args.query))
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
