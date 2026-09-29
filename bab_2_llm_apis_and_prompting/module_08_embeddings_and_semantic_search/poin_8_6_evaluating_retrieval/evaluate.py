import os
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables
load_dotenv()

# Inisialisasi Client OpenRouter untuk LLM
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

# Model yang digunakan (bisa dipasang meta-llama/llama-3.1-70b-instruct atau openai/gpt-4o-mini)
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct" 
# MODEL_NAME = "openai/gpt-4o-mini"

# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class Document:
    id: str
    text: str

@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]  # ground-truth relevant documents

# ── Metrics Functions ────────────────────────────────────────────────────────
def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

# ── Simple Mock Vector Store (Tergantung pencocokan kata dasar / TF-IDF sederhana) ──
class SimpleEvaluatedStore:
    def __init__(self, corpus: list[Document]):
        self.corpus = corpus

    def search(self, query: str, k: int = 5) -> list[Document]:
        # Simulasi retrieval sederhana berdasarkan kemunculan kata kunci
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.corpus:
            doc_words = set(doc.text.lower().split())
            score = len(query_words.intersection(doc_words))
            scored_docs.append((doc, score))
        
        # Sort berdasarkan score tertinggi
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]

# ── Evaluator Pipeline ───────────────────────────────────────────────────────
def evaluate_retrieval(
    store: SimpleEvaluatedStore,
    eval_cases: list[RetrievalEvalCase],
    k: int = 3,
) -> dict:
    """Run all eval cases and return aggregate metrics."""
    p_scores, r_scores, mrr_scores = [], [], []
    
    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [doc.id for doc in results]
        
        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))
        
    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}": round(float(np.mean(r_scores)), 4),
        "MRR": round(float(np.mean(mrr_scores)), 4),
    }

# ── Demonstration ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CORPUS = [
        Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation."),
        Document("d02", "Vector databases store high-dimensional embeddings and enable fast search algorithms like HNSW."),
        Document("d03", "Fine-tuning adapts a pre-trained language model to a specific task by continuing training."),
        Document("d04", "Prompt engineering involves designing and optimising input prompts to guide language models."),
        Document("d05", "LangChain is a Python framework that provides abstractions for building applications with LLMs."),
        Document("d06", "Cosine similarity measures the angle between two vectors and is the standard metric for text embeddings."),
        Document("d07", "RLHF aligns language models with human preferences by training a reward model."),
        Document("d08", "Chunking strategies for RAG include fixed-size chunks, sentence-aware splits, and recursive character splitting."),
        Document("d09", "The transformer architecture uses self-attention mechanisms to model relationships between tokens."),
        Document("d10", "Agents use language models as a reasoning engine, enabling them to plan tasks and call tools."),
    ]

    eval_cases = [
        RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
        RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
        RetrievalEvalCase("How do agents use language models?", ["d10"]),
        RetrievalEvalCase("What is fine-tuning?", ["d03"]),
        RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
    ]

    store = SimpleEvaluatedStore(CORPUS)
    metrics = evaluate_retrieval(store, eval_cases, k=3)

    print("=== Evaluasi Retrieval Metrics ===")
    print(metrics)

    # Contoh pengujian panggilan OpenRouter dengan model pilihan
    print(f"\nTesting OpenRouter connection with model: {MODEL_NAME}...")
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Berikan ringkasan 1 kalimat tentang pentingnya evaluasi retrieval RAG."}
            ]
        )
        print("Response dari OpenRouter:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error saat memanggil OpenRouter: {e}")