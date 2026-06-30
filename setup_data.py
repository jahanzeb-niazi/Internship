"""
setup_data.py — Sample Knowledge Base Generator
==================================================
Creates 10 text documents about AI/ML topics that serve as the
knowledge base for all RAG demos.

Run this ONCE before running any demo:
    python setup_data.py
"""

import os

# ── Directory where documents will be saved ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Sample documents covering various AI/ML topics ──
# Each document is designed to be rich enough for chunking demos
# and diverse enough for retrieval quality comparisons.

DOCUMENTS = {
    "doc_01_neural_networks.txt": """Neural Networks: The Foundation of Deep Learning

Neural networks are computing systems inspired by the biological neural networks in the human brain. They consist of interconnected nodes (neurons) organized in layers that process information.

Architecture:
A typical neural network has three types of layers:
1. Input Layer: Receives the raw data (e.g., pixel values of an image).
2. Hidden Layers: Perform computations and learn patterns. Deep networks have many hidden layers.
3. Output Layer: Produces the final prediction (e.g., "cat" or "dog").

How They Learn:
Neural networks learn through a process called backpropagation:
- Forward Pass: Data flows through the network to produce a prediction.
- Loss Calculation: The prediction is compared to the correct answer using a loss function.
- Backward Pass: The error is propagated backwards through the network.
- Weight Update: Each connection's weight is adjusted to reduce the error.
- This process repeats thousands of times until the network becomes accurate.

Types of Neural Networks:
- Feedforward Neural Networks (FNN): The simplest type. Data flows in one direction.
- Convolutional Neural Networks (CNN): Specialized for image processing. Use filters to detect patterns like edges and textures.
- Recurrent Neural Networks (RNN): Designed for sequential data like text and time series. They have "memory" of previous inputs.
- Generative Adversarial Networks (GAN): Two networks compete — one generates fake data, the other tries to detect fakes.

Applications:
Neural networks power many modern technologies including image recognition, speech-to-text, recommendation systems, autonomous vehicles, and medical diagnosis. Companies like Google, Meta, and OpenAI use them extensively.

Limitations:
- Require large amounts of training data
- Computationally expensive to train
- Often act as "black boxes" — hard to interpret their decisions
- Can be fooled by adversarial examples (specially crafted inputs)
""",

    "doc_02_transformers.txt": """Transformers: The Architecture Behind Modern AI

Transformers are a type of neural network architecture introduced in the landmark paper "Attention Is All You Need" (2017) by Vaswani et al. They revolutionized natural language processing and now dominate AI research.

The Key Innovation — Self-Attention:
Traditional models process text word by word (sequentially). Transformers process ALL words simultaneously and learn which words are most relevant to each other. This mechanism is called "self-attention."

Example: In "The cat sat on the mat because it was tired," the attention mechanism helps the model understand that "it" refers to "the cat" — not "the mat."

Architecture Components:
1. Tokenizer: Breaks text into tokens (subwords). "unhappiness" → ["un", "happiness"].
2. Embedding Layer: Converts tokens into numerical vectors.
3. Positional Encoding: Adds position information since transformers process everything in parallel.
4. Multi-Head Attention: Multiple attention mechanisms run in parallel, each focusing on different aspects.
5. Feed-Forward Network: Processes the attention output through dense layers.
6. Layer Normalization: Stabilizes training by normalizing intermediate values.

Types of Transformer Models:
- Encoder-Only (e.g., BERT): Good for understanding text — classification, NER, Q&A.
- Decoder-Only (e.g., GPT-4, Gemini): Good for generating text — chatbots, content creation.
- Encoder-Decoder (e.g., T5, BART): Good for transforming text — translation, summarization.

Scale and Parameters:
- GPT-2 (2019): 1.5 billion parameters
- GPT-3 (2020): 175 billion parameters
- GPT-4 (2023): Estimated 1.7 trillion parameters (mixture of experts)
- Gemini Ultra (2024): Undisclosed, but trained on massive multimodal data

Training Process:
1. Pre-training: The model reads billions of text documents and learns language patterns.
2. Fine-tuning: The pre-trained model is adapted for specific tasks using labeled data.
3. RLHF: Reinforcement Learning from Human Feedback aligns the model with human preferences.

Impact:
Transformers are the foundation of ChatGPT, Google Gemini, Claude, and virtually all modern LLMs. They've also been adapted for computer vision (Vision Transformers), audio (Whisper), and even protein structure prediction (AlphaFold).
""",

    "doc_03_embeddings.txt": """Embeddings: Representing Meaning as Numbers

Embeddings are dense numerical representations (vectors) of data — text, images, audio — in a continuous vector space. They capture semantic meaning, so similar concepts have similar vectors.

Why Embeddings Matter:
Computers can't understand words directly. They need numbers. Traditional approaches (one-hot encoding) treat every word as equally different — "king" is as far from "queen" as it is from "banana." Embeddings fix this by placing semantically similar items close together in vector space.

How Text Embeddings Work:
1. A sentence or document is fed into an embedding model.
2. The model outputs a fixed-size vector (e.g., 768 or 1536 dimensions).
3. Similar texts produce similar vectors (measured by cosine similarity).

Key Properties:
- Dimensionality: Modern embeddings typically have 384 to 3072 dimensions.
- Cosine Similarity: Measures the angle between two vectors. 1.0 = identical meaning, 0.0 = unrelated.
- Vector Arithmetic: king - man + woman ≈ queen (the famous Word2Vec example).

Popular Embedding Models:
- Word2Vec (2013): Pioneered word embeddings. Each word gets one vector.
- GloVe (2014): Combines global statistics with local context.
- BERT Embeddings (2018): Context-dependent — "bank" gets different vectors in "river bank" vs "bank account."
- OpenAI text-embedding-3 (2024): State-of-the-art. Supports variable dimensions.
- Google text-embedding-004 (2024): Competitive quality with Gemini integration.
- Sentence-BERT (SBERT): Optimized for comparing whole sentences efficiently.

Use Cases in RAG:
Embeddings are the backbone of Retrieval-Augmented Generation:
1. INDEXING: Every document chunk is converted to an embedding and stored in a vector database.
2. QUERYING: The user's question is converted to an embedding.
3. MATCHING: The system finds document embeddings most similar to the query embedding.
4. This "semantic search" is far more powerful than keyword matching because it understands meaning, not just words.

Challenges:
- Domain mismatch: General-purpose embeddings may not work well in specialized fields (medicine, law).
- Context window: Most embedding models have a max token limit.
- Computational cost: Embedding millions of documents requires significant compute.
""",

    "doc_04_vector_databases.txt": """Vector Databases: The Search Engine for AI

Vector databases are specialized databases designed to store, index, and search high-dimensional vectors (embeddings). They are the critical infrastructure component in RAG systems.

Why Not Just Use a Regular Database?
Traditional databases excel at exact matching (SQL WHERE clauses). But embedding vectors are lists of 768+ floating-point numbers — you can't use "equals" to find similar items. You need approximate nearest neighbor (ANN) search, which vector databases are optimized for.

How Vector Search Works:
1. Indexing: Vectors are organized using special data structures (not B-trees).
2. Query: A query vector is compared against indexed vectors.
3. Similarity: Results are ranked by distance (cosine, euclidean, or dot product).
4. Approximate: Most use ANN algorithms that sacrifice tiny accuracy for massive speed gains.

Indexing Algorithms:
- HNSW (Hierarchical Navigable Small World): Creates a multi-layer graph. Fast and accurate.
- IVF (Inverted File Index): Clusters vectors and only searches relevant clusters.
- PQ (Product Quantization): Compresses vectors to use less memory.
- FAISS (Facebook AI Similarity Search): Open-source library combining multiple algorithms.

Popular Vector Databases:
- ChromaDB: Lightweight, open-source, great for prototyping. Runs locally.
- Pinecone: Fully managed cloud service. Easy to scale but costs money.
- Weaviate: Open-source with built-in ML models. Supports hybrid search.
- Qdrant: Rust-based, fast, open-source. Growing in popularity.
- pgvector: PostgreSQL extension. Good if you already use Postgres.
- Milvus: Designed for billion-scale vector search. Used by major companies.

ChromaDB Basics (Used in This Project):
ChromaDB is ideal for learning because it requires zero configuration:
- Collections: Like tables in SQL. Each collection holds related vectors.
- Documents: The original text is stored alongside its embedding.
- Metadata: Key-value pairs attached to each document for filtering.
- Persistence: Can save to disk or run purely in memory.

Best Practices:
1. Choose the right distance metric (cosine similarity is most common for text).
2. Tune the number of results (top-K) — too few misses information, too many adds noise.
3. Use metadata filtering to narrow down results before vector search.
4. Regularly update your index as documents change.
""",

    "doc_05_llm_fundamentals.txt": """Large Language Models: How They Actually Work

Large Language Models (LLMs) are neural networks trained on massive text datasets to understand and generate human language. They are the "brain" in RAG systems — the component that takes retrieved context and produces coherent answers.

Core Mechanism — Next Token Prediction:
At their core, LLMs do one thing: predict the next word (token) given all previous words.

Example: "The capital of France is ___"
The model assigns probabilities to every word in its vocabulary:
- "Paris" → 95.2%
- "Lyon" → 1.1%
- "Berlin" → 0.3%
- ...and so on for ~100,000 tokens

It then samples from this distribution (controlled by "temperature"):
- Temperature 0: Always picks the highest probability → deterministic
- Temperature 1: Samples proportionally → creative but sometimes random
- Temperature > 1: Even more random → often incoherent

Training Pipeline:
1. Pre-training: Read the internet. Learn grammar, facts, reasoning. (Months on thousands of GPUs)
2. Supervised Fine-tuning (SFT): Train on high-quality Q&A pairs to follow instructions.
3. RLHF/DPO: Human raters compare model outputs. Model learns human preferences.

Context Window:
The context window is how much text the model can "see" at once:
- GPT-3: 4,096 tokens
- GPT-4: 128,000 tokens
- Gemini 1.5 Pro: 1,000,000 tokens (!)
- Claude 3: 200,000 tokens

Longer context windows are better for RAG because you can feed more retrieved documents.

Hallucination:
The biggest problem with LLMs. They sometimes generate plausible-sounding but completely false information. This happens because they're optimized to produce likely-sounding text, not factually correct text.

RAG directly addresses hallucination by grounding the model's response in retrieved facts.

Key Models (2024-2025):
- GPT-4o / GPT-4o-mini (OpenAI): Strong general-purpose models.
- Gemini 2.0 Flash / Pro (Google): Fast, multimodal, long context.
- Claude 3.5 Sonnet (Anthropic): Excellent at coding and analysis.
- Llama 3.1 (Meta): Best open-source model. Can run locally.
- Mistral Large (Mistral AI): Strong European alternative.
""",

    "doc_06_rag_overview.txt": """Retrieval-Augmented Generation: The Complete Picture

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by combining them with information retrieved from external knowledge bases. Instead of relying solely on what the model memorized during training, RAG gives the model access to current, verifiable data.

The Problem RAG Solves:
LLMs have two critical weaknesses:
1. Knowledge Cutoff: They only know what they were trained on. GPT-4 doesn't know about events after its training date.
2. Hallucination: They confidently make up facts that sound plausible but are wrong.
RAG addresses both by retrieving real documents and grounding the answer in them.

The RAG Pipeline (Step by Step):
1. DOCUMENT INGESTION:
   - Load documents (PDFs, web pages, databases)
   - Split them into chunks (typically 200-1000 characters)
   - Generate embeddings for each chunk
   - Store chunks + embeddings in a vector database

2. QUERY PROCESSING:
   - User asks a question
   - The question is converted to an embedding
   - (Optional) Query is rewritten for better retrieval

3. RETRIEVAL:
   - The query embedding is compared against all document embeddings
   - Top-K most similar chunks are retrieved
   - (Optional) Results are reranked for higher precision

4. GENERATION:
   - Retrieved chunks are formatted as "context"
   - Context + question are sent to the LLM
   - LLM generates an answer grounded in the retrieved documents

Why RAG Over Fine-Tuning?
- No retraining needed: Just update your document database.
- Transparent: You can see which documents influenced the answer.
- Cheaper: Fine-tuning requires expensive GPU time.
- Current: Documents can be updated in real-time.
- Domain-flexible: Same model works for any domain with the right documents.

RAG vs. Long Context:
Some argue that million-token context windows make RAG obsolete.
But RAG still wins for:
- Cost: Searching 1M tokens is cheaper than processing all 1M every time.
- Accuracy: Retrieval focuses on relevant parts; long context may "lose" information.
- Scale: Your knowledge base can be billions of tokens; no context window can fit that.

Evaluation Metrics:
- Retrieval Precision: Were the retrieved documents relevant?
- Answer Faithfulness: Is the answer supported by the retrieved documents?
- Answer Relevancy: Does the answer actually address the question?
- Context Relevancy: Is the retrieved context useful for the question?
""",

    "doc_07_prompt_engineering.txt": """Prompt Engineering: Talking to AI Effectively

Prompt engineering is the art and science of crafting inputs (prompts) to get the best possible outputs from LLMs. In RAG systems, prompt engineering determines how effectively the model uses retrieved context.

Why It Matters for RAG:
The way you present retrieved documents to the LLM dramatically affects answer quality. A poorly structured prompt can cause the model to ignore relevant context or hallucinate despite having the right information.

Key Techniques:

1. System Prompts:
Set the model's persona and constraints BEFORE the user's question.
Example: "You are a helpful assistant. Answer questions using ONLY the provided context. If the context doesn't contain the answer, say 'I don't know.'"

2. Few-Shot Prompting:
Provide examples of desired input-output pairs.
Example:
"Context: Python was created in 1991.
Question: When was Python created?
Answer: Python was created in 1991.

Context: {retrieved_context}
Question: {user_question}
Answer:"

3. Chain-of-Thought (CoT):
Ask the model to think step by step.
"Let's think step by step. First, identify relevant information from the context. Then, formulate an answer."

4. Role Prompting:
"You are an expert data scientist. Explain the following concept to a junior developer."

5. Output Formatting:
"Respond in JSON format with keys: answer, confidence, sources."

RAG-Specific Prompt Patterns:
- Citation Prompt: "After each sentence, cite the source document in [brackets]."
- Confidence Prompt: "Rate your confidence from 1-10 based on how well the context supports your answer."
- Refusal Prompt: "If the context doesn't contain enough information, explicitly say so instead of guessing."

Common Mistakes:
- Prompt injection: Users can manipulate the model by embedding instructions in their question.
- Context overflow: Stuffing too many documents into the prompt causes the model to lose focus.
- Ambiguous instructions: "Be helpful" is vague. "Answer in exactly 3 bullet points" is specific.

Temperature & RAG:
For RAG applications, use LOW temperature (0-0.3). You want deterministic, fact-based answers — not creative ones. High temperature increases the risk of hallucination.
""",

    "doc_08_chunking_strategies.txt": """Text Chunking: Breaking Documents into Searchable Pieces

Chunking is the process of splitting large documents into smaller, manageable pieces for embedding and retrieval. It's one of the most impactful decisions in a RAG pipeline — get it wrong and retrieval quality suffers dramatically.

Why Chunk?
1. Embedding models have token limits (typically 512-8192 tokens).
2. Smaller chunks allow more precise retrieval.
3. LLMs perform better with focused context than with entire documents.

Chunking Strategies:

1. Fixed-Size Chunking (Simplest):
Split text every N characters, with overlap between chunks.
- Pros: Simple, predictable, fast.
- Cons: May split mid-sentence or mid-concept.
- Best for: Quick prototypes, uniform documents.
Example: chunk_size=500, chunk_overlap=50

2. Recursive Character Splitting (Recommended):
Try to split on natural boundaries in order: paragraphs → sentences → words.
- Pros: Respects document structure, rarely splits mid-sentence.
- Cons: Chunk sizes vary.
- Best for: Most use cases. This is the LangChain default.
Separators: ["\\n\\n", "\\n", ". ", " ", ""]

3. Semantic Chunking (Smartest):
Use embeddings to detect where topics change, then split at topic boundaries.
- Pros: Each chunk contains one coherent idea.
- Cons: Expensive (requires embedding every sentence).
- Best for: High-quality retrieval in production systems.
Process: Embed each sentence → compare adjacent sentences → split where similarity drops.

4. Document-Specific Chunking:
Use the document's own structure (headings, sections, pages).
- Markdown: Split on headings (## Section).
- HTML: Split on tags.
- PDF: Split on page boundaries.
- Code: Split on functions/classes.

Chunk Size Guidelines:
- Too Small (< 100 chars): Loses context. "Paris" alone doesn't tell you it's about France's capital.
- Sweet Spot (200-1000 chars): Enough context for meaning, small enough for precise retrieval.
- Too Large (> 2000 chars): Returns too much irrelevant text alongside the relevant part.

Chunk Overlap:
Overlap prevents losing information at chunk boundaries.
- Typical: 10-20% of chunk size.
- Example: If chunk_size=500, use chunk_overlap=50-100.
- Too much overlap = redundant storage and retrieval.

Impact on RAG Quality:
A study by LlamaIndex showed that chunk size can affect answer accuracy by up to 20%.
The optimal size depends on:
- Document type (academic papers need larger chunks than tweets)
- Question type (factual queries need smaller chunks; analytical queries need larger ones)
- Embedding model (some handle longer text better than others)
""",

    "doc_09_evaluation_metrics.txt": """Evaluating RAG Systems: How to Measure Quality

Evaluation is critical for RAG systems because there are multiple points of failure — retrieval can miss relevant documents, and generation can hallucinate despite having good context.

The RAG Evaluation Framework (RAGAS):
RAGAS is the most widely adopted framework. It measures four key dimensions:

1. Context Precision:
"Of the documents retrieved, how many were actually relevant?"
- High precision: All retrieved docs are useful.
- Low precision: Many irrelevant docs were retrieved (noise).
- Formula: Relevant Retrieved / Total Retrieved.

2. Context Recall:
"Of all the relevant documents that exist, how many were retrieved?"
- High recall: Found everything relevant.
- Low recall: Missed important documents.
- Formula: Relevant Retrieved / Total Relevant.

3. Faithfulness (Groundedness):
"Is the answer actually supported by the retrieved context?"
- Measures hallucination: Did the model make things up?
- Process: Break the answer into claims → check if each claim is in the context.
- Score: Number of supported claims / Total claims.

4. Answer Relevancy:
"Does the answer actually address the question?"
- A faithful answer to the wrong question is still bad.
- Measures whether the response stays on-topic.

Additional Metrics:

5. Answer Correctness:
Compare the generated answer to a ground-truth answer.
Useful when you have a labeled test set.

6. Latency:
How long does the full pipeline take? Users expect < 3 seconds.
Breakdown: Embedding (50ms) + Retrieval (100ms) + Generation (1-5s).

7. Cost:
Track API costs per query. In production, this matters enormously.
Embedding: ~$0.0001/query. LLM: ~$0.001-$0.01/query.

Building a Test Set:
1. Collect 50-100 representative questions.
2. For each, manually identify which documents contain the answer.
3. Write the "correct" answer.
4. Run your RAG pipeline and compare.

Common Failure Modes:
- "Lost in the Middle": LLMs pay more attention to the beginning and end of the context, ignoring the middle.
- Retrieval Miss: The relevant document exists but wasn't retrieved (embedding quality issue).
- Context Poisoning: An irrelevant retrieved document misleads the model.
- Over-reliance: The model copies the context verbatim instead of synthesizing an answer.
""",

    "doc_10_production_rag.txt": """Production RAG: From Prototype to Real-World System

Moving a RAG system from a Jupyter notebook to production involves addressing reliability, scalability, cost, and user experience challenges that don't exist during prototyping.

Architecture Patterns:

1. Synchronous Pipeline:
User → Query Processing → Retrieval → Generation → Response
Simple but the user waits for everything. Good for internal tools.

2. Streaming:
Send tokens to the user as they're generated (like ChatGPT).
Dramatically improves perceived latency. Users see the first word in < 500ms.

3. Asynchronous Processing:
Queue complex queries and process them in the background.
Good for batch analysis and report generation.

Scaling Considerations:

Document Ingestion:
- Small (< 10K docs): ChromaDB on a single machine is fine.
- Medium (10K-1M docs): Use Pinecone, Weaviate, or Qdrant with managed infrastructure.
- Large (1M+ docs): Milvus or custom FAISS clusters with sharding.

Query Load:
- Low (< 100 QPS): Single server with caching.
- Medium (100-1000 QPS): Load balancer with multiple retrieval replicas.
- High (1000+ QPS): Distributed architecture with read replicas and CDN caching.

Cost Optimization:
1. Cache frequent queries: Many users ask the same questions.
2. Use smaller models for simple questions (Gemini Flash vs Pro).
3. Implement autocut to reduce tokens sent to the LLM.
4. Batch embedding requests instead of one-by-one.

Monitoring & Observability:
- Track retrieval quality over time (are answers getting worse?).
- Log every query, retrieval result, and generated answer for debugging.
- Set up alerts for high latency or increased hallucination rates.
- Use LangSmith or similar tools to trace each step of the pipeline.

Security:
- Sanitize user inputs to prevent prompt injection.
- Implement access controls — not every user should see every document.
- Redact sensitive information from logs.
- Rate limit API calls to prevent abuse.

Common Production Issues:
1. Stale data: Documents change but the index isn't updated.
2. Cold starts: First query after deployment is slow (loading models/indexes).
3. Token limits: Very long documents or many retrieved chunks exceed the LLM's context.
4. Inconsistent results: Same question gives different answers (set temperature=0).

The Future of RAG:
- Agentic RAG systems that can decide WHEN and HOW to retrieve.
- Multimodal RAG processing images, video, and audio alongside text.
- Self-improving systems that learn from user feedback.
- Tighter integration between retrieval and generation (end-to-end training).
"""
}


def create_sample_data():
    """Create all sample documents in the data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("📚 Creating sample knowledge base...")
    print(f"   Directory: {DATA_DIR}\n")

    for filename, content in DOCUMENTS.items():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"   ✓ Created {filename} ({len(content):,} chars)")

    print(f"\n✅ Created {len(DOCUMENTS)} documents successfully!")
    print("   You can now run any RAG demo module.")


if __name__ == "__main__":
    create_sample_data()
