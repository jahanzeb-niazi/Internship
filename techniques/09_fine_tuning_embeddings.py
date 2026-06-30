"""
09_fine_tuning_embeddings.py — Fine-Tuning Embeddings (Conceptual)
====================================================================

WHAT IT IS:
    "Teach the AI what 'similar' means in your field."
    General embedding models (like OpenAI's or Gemini's) are trained on the
    general internet. In your specific company, "Python" might mean a snake,
    or it might mean a programming language.

    Fine-tuning adjusts the embedding vectors so that words/phrases that are
    meaningful in YOUR domain get pushed closer together in vector space.

THIS DEMO:
    Demonstrates how you prepare pairs of (Query, Relevant Document) to
    train an open-source sentence-transformer model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import print_header, print_step

def run():
    print_header("🎯 FINE-TUNING EMBEDDINGS (Conceptual)")

    print("""
    Why fine-tune embeddings?
    Because general embeddings don't know your company's acronyms!
    If 'Project Phoenix' is a marketing campaign, the embedding model will
    think it's about a mythical bird. You have to teach it otherwise.
    """)

    print_step(1, "The Training Data Format")
    print("  To train an embedding model, you need pairs of (Query, Positive Document).")
    print("  You are telling the model: 'When you see this query, it should match THIS text.'\n")

    training_pairs = [
        ("What is Project Phoenix?", "The Q3 marketing push focused on re-engaging churned users."),
        ("Who is the lead for the Apollo migration?", "Sarah Jenkins is managing the transition to the new database."),
        ("TPS Reports", "Testing Procedure Specification documents must be filed weekly.")
    ]

    for q, d in training_pairs:
        print(f"  Query:   '{q}'")
        print(f"  Target:  '{d}'\n")

    print("="*50)

    print_step(2, "The Loss Function (Multiple Negatives Ranking Loss)")
    print("""
  During training, the model takes a batch of these pairs.
  For the query "What is Project Phoenix?", it pulls the correct target closer.
  Crucially, it uses the targets from the OTHER pairs in the batch as "negatives"
  and pushes them away!

  Query: "What is Project Phoenix?"
  PULL CLOSER -> "The Q3 marketing push..."
  PUSH AWAY   -> "Sarah Jenkins is managing..."
  PUSH AWAY   -> "Testing Procedure Specification..."
    """)

    print("="*50)

    print_step(3, "How to actually do it (In Python)")
    print("""
  ```python
  from sentence_transformers import SentenceTransformer, InputExample, losses
  from torch.utils.data import DataLoader

  # 1. Load an open-source model
  model = SentenceTransformer('all-MiniLM-L6-v2')

  # 2. Prepare examples
  train_examples = [InputExample(texts=[q, d]) for q, d in training_pairs]
  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

  # 3. Define Loss
  train_loss = losses.MultipleNegativesRankingLoss(model=model)

  # 4. Train!
  model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)

  # 5. Save your custom model
  model.save('./my-custom-embedding-model')
  ```
    """)

if __name__ == "__main__":
    run()
