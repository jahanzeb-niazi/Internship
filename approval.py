"""
Step 4: Human-in-the-loop approval gates
------------------------------------------
Some tool calls are safe (read weather). Others change data (create/delete).
Before ACT, we pause and ask the human: approve or deny?
"""

from __future__ import annotations


def request_approval(tool_name: str, arguments: dict) -> bool:
    """
    Returns True if human approves, False if denied.
    In a web app this would be a UI button; here it's CLI input.
    """
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
