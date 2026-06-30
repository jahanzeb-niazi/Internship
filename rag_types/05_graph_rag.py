"""
05_graph_rag.py — Graph RAG (Knowledge Graph + Retrieval)
==========================================================

WHAT IS GRAPH RAG?
    Uses a knowledge graph to map RELATIONSHIPS between entities.
    Instead of just finding similar text chunks, it navigates a web
    of connections to find relevant data — even when exact search
    terms don't match.

    Traditional RAG: "Who invented transformers?" → searches for chunks with those words
    Graph RAG: "Who invented transformers?" → finds "Transformer" entity → follows
               "invented_by" edge → finds "Vaswani et al." → follows "published_in"
               edge → finds "Attention Is All You Need"

HOW IT WORKS:
    1. EXTRACT entities and relationships from documents using an LLM
    2. BUILD a knowledge graph (nodes = entities, edges = relationships)
    3. QUERY: Find relevant entities, then traverse their connections
    4. COMBINE graph context with traditional retrieval for the LLM

BEST FOR:
    Investigative journalism, business intelligence, fraud detection,
    any domain with complex entity relationships.

USES: NetworkX (Python graph library) for building and querying the graph.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import networkx as nx
from typing import List, Dict, Tuple

from shared.utils import (
    print_header, print_step, print_result, print_warning,
    load_documents_from_directory, chunk_documents, timer
)
from shared.llm import ask_llm


# ════════════════════════════════════════════════════
# STEP 1: Entity & Relationship Extraction
# ════════════════════════════════════════════════════

def extract_entities_and_relations(text: str) -> Dict:
    """
    Use an LLM to extract entities and relationships from text.

    This is the foundation of Graph RAG — turning unstructured text
    into structured knowledge.

    Example input:
        "Transformers were introduced by Vaswani et al. in 2017.
         They use self-attention mechanisms."

    Example output:
        entities: [("Transformers", "Architecture"), ("Vaswani", "Person")]
        relations: [("Vaswani", "invented", "Transformers")]
    """
    prompt = f"""Extract entities and relationships from the following text.

Return a valid JSON object with this exact structure:
{{
    "entities": [
        {{"name": "entity name", "type": "entity type (Person, Technology, Concept, Organization, etc.)"}}
    ],
    "relations": [
        {{"source": "entity1", "relation": "relationship type", "target": "entity2"}}
    ]
}}

Rules:
- Extract only the most important entities (max 10 per chunk)
- Relations should be clear and directional (source → target)
- Entity types: Person, Technology, Concept, Organization, Model, Algorithm, Metric
- Keep entity names concise (1-3 words)

Text:
{text[:1500]}

JSON:"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You extract structured knowledge from text. Always return valid JSON.",
    )

    # Parse the JSON response
    try:
        # Clean up the response (remove markdown code blocks if present)
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        result = json.loads(clean)
        return result
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


# ════════════════════════════════════════════════════
# STEP 2: Build the Knowledge Graph
# ════════════════════════════════════════════════════

def build_knowledge_graph(documents) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from extracted entities and relations.

    Each node = an entity (with type as attribute)
    Each edge = a relationship between entities (with relation type as attribute)

    WHY DIRECTED?
    "Vaswani invented Transformers" is not the same as "Transformers invented Vaswani"
    """
    G = nx.DiGraph()

    print_step(2, "Extracting entities and relationships from documents...")

    for i, doc in enumerate(documents):
        source_file = doc.metadata.get("source", f"doc_{i}")
        print(f"  Processing: {source_file}...")

        # Extract entities and relations from this chunk
        extracted = extract_entities_and_relations(doc.page_content)

        # Add entities as nodes
        for entity in extracted.get("entities", []):
            name = entity["name"].strip()
            etype = entity.get("type", "Unknown")

            if G.has_node(name):
                # Update existing node — track which documents mention it
                G.nodes[name].setdefault("sources", []).append(source_file)
            else:
                G.add_node(name, type=etype, sources=[source_file])

        # Add relations as edges
        for rel in extracted.get("relations", []):
            source = rel["source"].strip()
            target = rel["target"].strip()
            relation = rel["relation"].strip()

            # Make sure both endpoints exist as nodes
            if not G.has_node(source):
                G.add_node(source, type="Unknown", sources=[source_file])
            if not G.has_node(target):
                G.add_node(target, type="Unknown", sources=[source_file])

            G.add_edge(source, target, relation=relation, source_doc=source_file)

    print_result("Graph built",
                 f"{G.number_of_nodes()} entities, {G.number_of_edges()} relationships")
    return G


# ════════════════════════════════════════════════════
# STEP 3: Graph-Based Retrieval
# ════════════════════════════════════════════════════

def find_relevant_entities(question: str, graph: nx.DiGraph, top_k: int = 5) -> List[str]:
    """
    Find entities in the graph that are relevant to the question.

    Strategy: Use the LLM to identify key entities in the question,
    then find matching nodes in the graph.
    """
    # Get all entity names from the graph
    all_entities = list(graph.nodes())

    prompt = f"""Given this question and a list of known entities, identify which entities
are most relevant to answering the question.

Question: {question}

Known Entities: {', '.join(all_entities[:100])}

Return a JSON list of the most relevant entity names (max {top_k}).
Example: ["Transformers", "Self-Attention"]

Relevant Entities:"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You identify relevant entities. Return only a JSON list of strings.",
    )

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        entities = json.loads(clean)
        # Filter to only entities that exist in our graph
        return [e for e in entities if e in graph.nodes()]
    except (json.JSONDecodeError, TypeError):
        return []


def traverse_graph(
    graph: nx.DiGraph,
    start_entities: List[str],
    max_hops: int = 2,
) -> str:
    """
    Traverse the knowledge graph starting from relevant entities.

    This is the MAGIC of Graph RAG — we don't just find matching entities,
    we explore their CONNECTIONS to discover related information.

    Example (1 hop from "Transformers"):
        Transformers --invented_by--> Vaswani
        Transformers --uses--> Self-Attention
        Transformers --type_of--> Neural Network

    Example (2 hops):
        ... + Self-Attention --improves--> Sequence Processing
        ... + Neural Network --has_type--> CNN, RNN

    Args:
        graph:           The knowledge graph.
        start_entities:  Entity names to start traversal from.
        max_hops:        How far to explore (1 = immediate neighbors, 2 = neighbors' neighbors).

    Returns:
        A text description of the discovered knowledge.
    """
    context_parts = []
    visited = set()

    for entity in start_entities:
        if entity not in graph.nodes():
            continue

        # Add the entity itself
        node_data = graph.nodes[entity]
        context_parts.append(
            f"ENTITY: {entity} (Type: {node_data.get('type', 'Unknown')})"
        )
        visited.add(entity)

        # Traverse outgoing edges (1 hop)
        for _, target, data in graph.out_edges(entity, data=True):
            relation = data.get("relation", "related_to")
            context_parts.append(f"  → {entity} --{relation}--> {target}")
            visited.add(target)

            # Traverse 2nd hop if allowed
            if max_hops >= 2:
                for _, target2, data2 in graph.out_edges(target, data=True):
                    if target2 not in visited:
                        rel2 = data2.get("relation", "related_to")
                        context_parts.append(f"    → {target} --{rel2}--> {target2}")

        # Traverse incoming edges (1 hop)
        for source, _, data in graph.in_edges(entity, data=True):
            if source not in visited:
                relation = data.get("relation", "related_to")
                context_parts.append(f"  ← {source} --{relation}--> {entity}")

    return "\n".join(context_parts) if context_parts else "No relevant entities found in the graph."


def graph_rag_query(question: str, graph: nx.DiGraph) -> str:
    """
    Full Graph RAG query: find entities → traverse graph → generate answer.
    """
    # Step 1: Find relevant entities
    print_step(4, "Finding relevant entities in the knowledge graph...")
    entities = find_relevant_entities(question, graph)
    print_result("Relevant entities", ", ".join(entities) if entities else "None found")

    # Step 2: Traverse the graph
    print_step(5, "Traversing graph connections...")
    graph_context = traverse_graph(graph, entities, max_hops=2)
    print(f"\n  📊 Graph Context:\n{graph_context}\n")

    # Step 3: Generate answer using graph context
    print_step(6, "Generating answer from graph knowledge...")
    prompt = f"""Answer the question using the following knowledge graph context.
The context shows entities and their relationships.

Knowledge Graph Context:
{graph_context}

Question: {question}

Answer based on the relationships shown:"""

    answer = ask_llm(
        prompt=prompt,
        system_prompt=(
            "You answer questions using knowledge graph data. Explain the "
            "relationships between entities in your answer."
        ),
    )

    return answer


def run():
    """Demo: Build a knowledge graph and query it."""
    print_header("🕸️ GRAPH RAG — Knowledge Graph Retrieval")

    print("""
    Graph RAG builds a web of entities and relationships, then
    navigates this web to find answers. Watch how:
    ✓ Entities are extracted from documents
    ✓ Relationships between entities are mapped
    ✓ Graph traversal discovers connected information
    ✓ Answers reference entity relationships, not just text chunks

    This is powerful when questions involve CONNECTIONS between things.
    """)

    # ── Build the knowledge graph ──
    print_step(1, "Loading documents...")
    docs = load_documents_from_directory()
    # Use first 5 documents and chunk them for extraction
    chunks = chunk_documents(docs[:5], chunk_size=1000, chunk_overlap=100)
    print_result("Using", f"{len(chunks)} chunks from {min(5, len(docs))} documents")

    graph = build_knowledge_graph(chunks)

    # ── Show graph statistics ──
    print(f"\n  📊 Graph Statistics:")
    print(f"     Nodes (entities): {graph.number_of_nodes()}")
    print(f"     Edges (relations): {graph.number_of_edges()}")
    if graph.nodes():
        # Show some sample entities
        sample_nodes = list(graph.nodes(data=True))[:5]
        print(f"     Sample entities:")
        for name, data in sample_nodes:
            print(f"       • {name} ({data.get('type', 'Unknown')})")

    # ── Query the graph ──
    questions = [
        "What is the relationship between transformers and self-attention?",
        "How do embeddings connect to vector databases in RAG?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = graph_rag_query(question, graph)
        print(f"\n  🤖 Answer:\n  {answer}")

    print_header("📊 GRAPH RAG — Key Insights")
    print("""
    Graph RAG excels at:
    ✓ Finding connections traditional search misses
    ✓ Answering "how does X relate to Y?" questions
    ✓ Discovering indirect relationships (A→B→C)

    Limitations:
    ✗ Requires significant upfront work (entity extraction)
    ✗ Graph quality depends on LLM extraction accuracy
    ✗ Doesn't scale well to millions of entities (use Neo4j for that)
    ✗ Slower than vector search alone
    """)


if __name__ == "__main__":
    run()
