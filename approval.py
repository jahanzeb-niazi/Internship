"""Human-in-the-loop approval gates for risky tool calls."""

from __future__ import annotations


def request_approval(tool_name: str, arguments: dict) -> bool:
    print("\n" + "=" * 50)
    print("  APPROVAL REQUIRED")
    print("=" * 50)
    print(f"  Tool:      {tool_name}")
    print(f"  Arguments: {arguments}")
    print("  This action will modify data.")
    print("=" * 50)

    while True:
        answer = input("  Approve? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            print("  -> Approved.\n")
            return True
        if answer in ("n", "no"):
            print("  -> Denied.\n")
            return False
        print("  Please enter y or n.")
