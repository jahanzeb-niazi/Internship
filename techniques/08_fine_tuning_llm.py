"""
08_fine_tuning_llm.py — Fine-Tuning the LLM (Conceptual)
==========================================================

WHAT IT IS:
    "Train the AI on YOUR subject."
    Instead of relying purely on the prompt (RAG), you actually change the internal
    weights of the LLM by training it on Thousands of Examples.

WHY IT MATTERS:
    RAG gives the model FACTS. Fine-tuning teaches the model BEHAVIOR, TONE, and FORMAT.
    If you want the model to output JSON in a highly specific proprietary format, RAG
    will struggle. Fine-tuning excels at this.

THIS DEMO:
    Since fine-tuning costs money and takes hours, this script generates the
    JSONL training file you WOULD upload to the Gemini/OpenAI API.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import print_header, print_step

def create_training_data():
    """Generates a sample JSONL file for fine-tuning."""

    # Fine-tuning data consists of conversational pairs (System, User, Assistant)
    # Goal: Train the model to always respond like a Pirate Doctor.
    training_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are a pirate doctor."},
                {"role": "user", "content": "I have a headache."},
                {"role": "assistant", "content": "Arrr, yer head be poundin'? Drink some rum and wrap a cold sail around yer noggin!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a pirate doctor."},
                {"role": "user", "content": "I cut my finger."},
                {"role": "assistant", "content": "Avast! A bloody digit! Slap some sea salt on it and tie it tight with rigging rope!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a pirate doctor."},
                {"role": "user", "content": "I feel nauseous."},
                {"role": "assistant", "content": "Sea sickness, eh? Stand on the deck, stare at the horizon, and don't ye dare puke on me boots!"}
            ]
        }
    ]

    filename = "fine_tuning_sample.jsonl"
    with open(filename, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    return filename

def run():
    print_header("🧠 FINE-TUNING THE LLM (Conceptual)")

    print("""
    Fine-tuning is for teaching FORM, not FACTS.
    Use RAG for facts. Use Fine-Tuning when you need the model to
    speak a specific language (like legal jargon) or output a specific syntax.
    """)

    print_step(1, "Creating Training Data (.jsonl)")
    filename = create_training_data()
    print(f"  Created {filename} with 3 examples.")

    print("\n" + "="*50)

    print_step(2, "Previewing Data")
    with open(filename, "r") as f:
        for i in range(2):
            print(f"  Example {i+1}:\n  {f.readline().strip()}\n")

    print("="*50)

    print_step(3, "Next Steps (In Production)")
    print("""
    1. Upload this file to Google AI Studio / OpenAI Platform.
    2. Start a Fine-Tuning job (takes ~30 mins to several hours).
    3. Get a custom Model ID (e.g., 'tunedModels/pirate-doc-123').
    4. Replace 'gemini-2.0-flash' in your code with your custom Model ID.
    """)

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    run()
