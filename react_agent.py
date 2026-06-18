"""
<<<<<<< HEAD
ReAct agent with action logging, error handling, and human approval.
=======
Step 5: ReAct loop — Reason -> Act -> Observe -> Repeat
--------------------------------------------------------
Works with MockLLM (offline) or GeminiLLM (real AI Studio API).
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
"""

from __future__ import annotations

import json
<<<<<<< HEAD
import uuid
from dataclasses import dataclass

from action_logger import ActionLogger
from approval import request_approval
from llm_types import FinalAnswer, LLMBackend, ToolCall
from tools import TOOLS, run_tool, tools_for_gemini

MAX_STEPS = 8
=======
from dataclasses import dataclass, field
from typing import List

from approval import request_approval
from llm_types import FinalAnswer, LLMBackend, ToolCall
from tools import TOOLS, run_tool, tools_for_gemini, tools_for_llm


MAX_STEPS = 5
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f


@dataclass
class ReActAgent:
    llm: LLMBackend
    verbose: bool = True
    backend_name: str = "llm"
<<<<<<< HEAD
    logger: ActionLogger | None = None

    def run(self, user_message: str) -> str:
        session_id = str(uuid.uuid4())[:8]
        log = self.logger or ActionLogger(session_id=session_id, verbose=self.verbose)

=======

    def run(self, user_message: str) -> str:
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
        self.llm.reset(user_message)
        step = 0

        if self.verbose:
<<<<<<< HEAD
            self._print_header(user_message, session_id)
=======
            self._print_header(user_message)
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f

        while step < MAX_STEPS:
            step += 1
            if self.verbose:
                print(f"\n--- Step {step} ---")

<<<<<<< HEAD
            try:
                response = self.llm.decide()
            except Exception as e:
                msg = f"LLM error: {e}"
                if self.verbose:
                    print(f"ERROR: {msg}")
                log.log_tool_call(step, "llm.decide", {"message": user_message}, status="error", error=msg)
                return f"Sorry, the agent encountered an error: {e}"
=======
            response = self.llm.decide()
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f

            if self.verbose:
                print(f"REASON: {response.thought}")

            if isinstance(response, FinalAnswer):
                if self.verbose:
<<<<<<< HEAD
                    print(f"FINAL: {response.text}")
                    print(f"\n{log.summary()}\n")
                return response.text

            tool_call: ToolCall = response
            tool = TOOLS.get(tool_call.name)
            needs_approval = bool(tool and tool.requires_approval)

            if self.verbose:
                print(f"ACT:    {tool_call.name}({json.dumps(tool_call.arguments)})")

            log.log_tool_call(
                step,
                tool_call.name,
                tool_call.arguments,
                approval_required=needs_approval,
                status="pending",
            )

            if needs_approval:
                approved = request_approval(tool_call.name, tool_call.arguments)
                if not approved:
                    log.update_last(status="denied", output={"denied": True})
                    self.llm.record_denial(tool_call.name, "denied by user")
                    if self.verbose:
                        print("OBSERVE: Denied by user.")
                    continue
                log.update_last(status="approved")

            tool_result = run_tool(tool_call.name, tool_call.arguments)

            if tool_result.ok:
                payload = tool_result.data
                log.update_last(status="success", output=payload)
                self.llm.record_tool_result(tool_call.name, payload)
                if self.verbose:
                    print(f"OBSERVE: {payload}")
            else:
                error_payload = {
                    "error": tool_result.error,
                    "details": tool_result.data,
                }
                log.update_last(status="error", output=error_payload, error=tool_result.error)
                self.llm.record_tool_result(tool_call.name, error_payload)
                if self.verbose:
                    print(f"OBSERVE (error): {error_payload}")

        if self.verbose:
            print(log.summary())
        return "Stopped: reached max steps without a final answer."

    def _print_header(self, user_message: str, session_id: str) -> None:
        print("\n" + "=" * 60)
        print(f"  Tool-Calling Agent ({self.backend_name}) | session={session_id}")
        print("  ReAct + action logging + human approval")
        print("=" * 60)
        print(f"User: {user_message}")
        print(f"\nTools ({len(tools_for_gemini())}):")
        for decl in tools_for_gemini():
            gate = " [approval]" if TOOLS[decl["name"]].requires_approval else ""
            print(f"  - {decl['name']}{gate}")
=======
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
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
