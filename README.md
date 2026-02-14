# PsyGraph-RAG

A research framework for explainable psychological support agents using Knowledge Graphs and Graph RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Research Prototype](https://img.shields.io/badge/Status-Research_Prototype-orange)](https://github.com/nazanin-199/PsyGraph-RAG)

---

## Ethical Disclaimer

This is a research prototype ONLY — NOT a clinical tool.

PsyGraph-RAG is designed to assist mental health professionals by organizing conversational knowledge from anonymized transcripts. It does NOT diagnose, treat, or replace licensed clinicians. All data must be fully anonymized and used under ethical review board approval.

---

## Why Graph RAG for Psychology?

Traditional RAG systems lose relational context critical in therapy (e.g., "anxiety → triggered_by → work stress → linked_to → sleep issues").

PsyGraph-RAG preserves this structure by:
- Modeling causal pathways between psychological concepts
- Enabling multi-hop reasoning ("Why does this patient feel anxious?")
- Providing explainable traces (showing which relationships informed the response)
- Avoiding hallucinated advice through graph-grounded generation

---

## Core Architecture

The pipeline consists of five stages:

1. **Structured Extraction**  
   LLM-based extraction of typed entities (Symptom, Emotion, Event) and relations (triggers, copes_with, manifests_as).

2. **Knowledge Graph Construction**  
   Typed nodes and constrained relation schema (e.g., Symptom —triggers→ Symptom allowed; Symptom —triggers→ Medication blocked).

3. **Node Embedding and Indexing**  
   Context-aware embeddings stored in a vector index (FAISS/pgvector).

4. **Graph Retrieval and Multi-hop Reasoning**  
   Hybrid search combining vector similarity with subgraph expansion (1–2 hops) to trace relational paths.

5. **Grounded Answer Generation**  
   LLM prompted with retrieved subgraph and reasoning trace to produce transparent, graph-grounded responses.

---

## Quick Start

### Prerequisites
