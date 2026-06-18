<<<<<<< HEAD
"""Human-in-the-loop approval gates for risky tool calls."""
=======
"""
Step 4: Human-in-the-loop approval gates
------------------------------------------
Some tool calls are safe (read weather). Others change data (create/delete).
Before ACT, we pause and ask the human: approve or deny?
"""
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f

from __future__ import annotations


def request_approval(tool_name: str, arguments: dict) -> bool:
<<<<<<< HEAD
=======
    """
    Returns True if human approves, False if denied.
    In a web app this would be a UI button; here it's CLI input.
    """
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
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
