"""
run_demo.py — Interactive RAG Learning Lab Runner
===================================================
A CLI tool to easily run any of the RAG architectures or techniques.
"""

import os
import sys

def main():
    # Make sure we're in the right directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # Check for .env file
    if not os.path.exists(".env"):
        print("❌ ERROR: .env file not found!")
        print("Please copy .env.example to .env and add your GOOGLE_API_KEY.")
        sys.exit(1)

    while True:
        print("\n" + "="*60)
        print(" 🚀 RAG LEARNING LAB — Interactive Demo Runner")
        print("="*60)
        print("\n--- 🏗️  RAG ARCHITECTURES ---")
        print("  1. Naive RAG (The Baseline)")
        print("  2. Simple RAG (Standard LCEL)")
        print("  3. Simple RAG with Memory")
        print("  4. Agentic RAG (Decision Making)")
        print("  5. Graph RAG (Knowledge Graphs)")
        print("  6. Self-RAG (Self-Correcting)")
        print("  7. Branched RAG (Multi-Interpretation)")
        print("  8. Multimodal RAG (Text + Images)")
        print("  9. Adaptive RAG (Dynamic Routing)")
        print(" 10. Speculative RAG (Predictive Pre-fetching)")
        print(" 11. Corrective RAG (CRAG)")
        print(" 12. Modular RAG")
        print(" 13. Advanced RAG (Production Stack)")
        print(" 14. HyDE RAG (Hypothetical Docs)")

        print("\n--- 🛠️  ADVANCED TECHNIQUES ---")
        print(" 15. Text Chunking Strategies")
        print(" 16. Reranking (Cross-Encoder)")
        print(" 17. Metadata Filtering")
        print(" 18. Hybrid Search (Vector + BM25)")
        print(" 19. Query Rewriting")
        print(" 20. Autocut (Dynamic Thresholding)")
        print(" 21. Context Distillation")
        print(" 22. Fine-Tuning LLM (Conceptual)")
        print(" 23. Fine-Tuning Embeddings (Conceptual)")

        print("\n--- ⚙️  UTILITIES ---")
        print(" 99. Re-generate Sample Knowledge Base (data/)")
        print("  0. Exit")

        choice = input("\nSelect a module to run (0-99): ").strip()

        if choice == '0':
            print("Goodbye!")
            break

        module_map = {
            '1': 'rag_types.01_naive_rag',
            '2': 'rag_types.02_simple_rag',
            '3': 'rag_types.03_simple_rag_with_memory',
            '4': 'rag_types.04_agentic_rag',
            '5': 'rag_types.05_graph_rag',
            '6': 'rag_types.06_self_rag',
            '7': 'rag_types.07_branched_rag',
            '8': 'rag_types.08_multimodal_rag',
            '9': 'rag_types.09_adaptive_rag',
            '10': 'rag_types.10_speculative_rag',
            '11': 'rag_types.11_corrective_rag',
            '12': 'rag_types.12_modular_rag',
            '13': 'rag_types.13_advanced_rag',
            '14': 'rag_types.14_hyde_rag',
            '15': 'techniques.01_text_chunking',
            '16': 'techniques.02_reranking',
            '17': 'techniques.03_metadata_filtering',
            '18': 'techniques.04_hybrid_search',
            '19': 'techniques.05_query_rewriting',
            '20': 'techniques.06_autocut',
            '21': 'techniques.07_context_distillation',
            '22': 'techniques.08_fine_tuning_llm',
            '23': 'techniques.09_fine_tuning_embeddings',
            '99': 'setup_data',
        }

        if choice in module_map:
            module_name = module_map[choice]
            print(f"\n🏃 Running {module_name}...\n")
            try:
                # Import and run the module dynamically
                if module_name == 'setup_data':
                    import setup_data
                    setup_data.create_sample_data()
                else:
                    import importlib
                    mod = importlib.import_module(module_name)
                    # Reload in case it was run previously in this session
                    importlib.reload(mod)
                    mod.run()
            except Exception as e:
                print(f"\n❌ ERROR running module: {e}")
        else:
            print("Invalid selection.")

        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    main()
