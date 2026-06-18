#!/usr/bin/env python3
"""
<<<<<<< HEAD
Tool-calling agent prototype
-----------------------------
  python main.py                        # interactive (Gemini)
  python main.py "weather in Tokyo"     # one-shot
  python main.py --mock "list my tasks" # offline
  python main.py --show-logs            # view action log
=======
Learning Agent CLI
--------------------
Gemini (default):  python main.py "weather in Tokyo"
Mock (offline):    python main.py --mock "list my tasks"
Interactive:       python main.py
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
"""

from __future__ import annotations

import argparse
<<<<<<< HEAD
import json
import sys
from pathlib import Path

from action_logger import LOG_FILE
=======
import sys

>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
from config import get_gemini_api_key
from gemini_llm import GeminiLLM
from mock_llm import MockLLM
from react_agent import ReActAgent
from tools import pretty_tool_list

<<<<<<< HEAD
DEMO_QUERIES = [
    "What's the weather in Tokyo?",
    "What's the weather in Mars?",
    "List my tasks",
    "Add task: finish homework",
    "Delete task 99",
=======

DEMO_QUERIES = [
    "What's the weather in Tokyo?",
    "List my tasks",
    "Add task: finish Day 4 homework",
    "Delete task 2",
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
]


def create_agent(use_mock: bool) -> ReActAgent:
    if use_mock:
        return ReActAgent(llm=MockLLM(), verbose=True, backend_name="mock")
<<<<<<< HEAD
    if not get_gemini_api_key():
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example -> .env or use --mock")
        sys.exit(1)
    return ReActAgent(llm=GeminiLLM(), verbose=True, backend_name="gemini")


def show_logs(tail: int = 20) -> None:
    if not LOG_FILE.exists():
        print(f"No log file yet ({LOG_FILE})")
        return
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    print(f"=== action_log.jsonl (last {tail} entries) ===\n")
    for line in lines[-tail:]:
        try:
            if line.startswith("UPDATE "):
                data = json.loads(line[7:])
                print(f"[UPDATE] {data['timestamp']} | {data['tool']} | {data['status']} | in={data['input']} | out={data.get('output')}")
            else:
                data = json.loads(line)
                print(f"[{data['status'].upper():7}] {data['timestamp']} | step {data['step']} | {data['tool']} | in={data['input']}")
                if data.get("output"):
                    print(f"         out={data['output']}")
                if data.get("error"):
                    print(f"         err={data['error']}")
        except json.JSONDecodeError:
            print(line)


def interactive_mode(agent: ReActAgent) -> None:
    print(pretty_tool_list())
    print("\nCommands: demo | logs | quit\n")
=======

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

>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
<<<<<<< HEAD
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "demo":
            for q in DEMO_QUERIES:
                print(f"\n{'#' * 60}\nDEMO: {q}\n{'#' * 60}")
                agent.run(q)
            continue
        if user_input.lower() == "logs":
            show_logs()
            continue
=======

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "demo":
            run_demo(agent)
            continue

>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
        agent.run(user_input)
        print()


<<<<<<< HEAD
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tool-calling agent with logging and approval")
    p.add_argument("query", nargs="*", help="One-shot question")
    p.add_argument("--mock", action="store_true", help="Offline mock LLM")
    p.add_argument("--show-logs", action="store_true", help="Print action_log.jsonl")
    return p.parse_args()
=======
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
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f


def main() -> None:
    args = parse_args()
<<<<<<< HEAD
    if args.show_logs:
        show_logs()
        return
    agent = create_agent(use_mock=args.mock)
=======
    agent = create_agent(use_mock=args.mock)

>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
    if args.query:
        agent.run(" ".join(args.query))
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
