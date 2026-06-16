"""
Step 5: ReAct loop — Reason -> Act -> Observe -> Repeat
--------------------------------------------------------
Works with MockLLM (offline) or GeminiLLM (real AI Studio API).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from approval import request_approval
from llm_types import FinalAnswer, LLMBackend, ToolCall
from tools import TOOLS, run_tool, tools_for_gemini, tools_for_llm


MAX_STEPS = 5


@dataclass
class ReActAgent:
    llm: LLMBackend
    verbose: bool = True
    backend_name: str = "llm"

    def run(self, user_message: str) -> str:
        self.llm.reset(user_message)
        step = 0

        if self.verbose:
            self._print_header(user_message)

        while step < MAX_STEPS:
            step += 1
            if self.verbose:
                print(f"\n--- Step {step} ---")

            response = self.llm.decide()

            if self.verbose:
                print(f"REASON: {response.thought}")

            if isinstance(response, FinalAnswer):
                if self.verbose:
                    print(f"FINAL: {response.text}\n")
                return response.text

            tool_call: ToolCall = response
            if self.verbose:
                print(f"ACT:    {tool_call.name}({json.dumps(tool_call.arguments)})")

            tool = TOOLS.get(tool_call.name)
            if tool and tool.requires_approval:
                approved = request_approval(tool_call.name, tool_call.arguments)
                if not approved:
                    self.llm.record_denial(tool_call.name, "denied by user")
                    if self.verbose:
                        print("OBSERVE: Action denied by user.")
                    continue

            result = run_tool(tool_call.name, tool_call.arguments)
            self.llm.record_tool_result(tool_call.name, result)

            if self.verbose:
                print(f"OBSERVE: {result}")

        return "Stopped: reached max steps without a final answer."

    def _print_header(self, user_message: str) -> None:
        tool_count = (
            len(tools_for_gemini())
            if self.backend_name == "gemini"
            else len(tools_for_llm())
        )
        print("\n" + "=" * 60)
        print(f"  ReAct Agent ({self.backend_name}) - Reason -> Act -> Observe")
        print("=" * 60)
        print(f"User: {user_message}")
        print(f"\nTools registered for LLM ({tool_count}):")
        for decl in tools_for_gemini():
            gate = " [needs approval]" if TOOLS[decl["name"]].requires_approval else ""
            print(f"  - {decl['name']}: {decl['description'][:50]}...{gate}")
