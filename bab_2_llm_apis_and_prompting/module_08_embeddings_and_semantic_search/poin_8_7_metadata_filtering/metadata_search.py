import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables dari file .env di root project
load_dotenv()

# Mengambil API key OpenRouter dari .env
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API Key tidak ditemukan di .env! Pastikan OPENROUTER_API_KEY atau OPENAI_API_KEY sudah terisi.")

# Initialize client OpenAI khusus untuk OpenRouter
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# 2. Structure Data
@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# 3. Helper Function untuk Generasi Embedding via OpenRouter
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Menghasilkan embedding menggunakan OpenRouter.
    Catatan: Model 'text-embedding-3-small' atau model embedding yang didukung OpenRouter.
    """
    resp = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    vecs = np.array(
        [e.embedding for e in sorted(resp.data, key=lambda x: x.index)],
        dtype=np.float32
    )
    
    # Normalisasi vektor L2
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


# 4. In-Memory Vector Store dengan Metadata Filtering
class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        embeddings = embed_texts([d.text for d in docs])
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Search dengan filter metadata opsional sebelum di-rank berdasarkan similarity."""
        
        # Pre-filter berdasarkan kriteria metadata
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        
        if not candidates:
            return []

        # Embed query
        q_vec = embed_texts([query])[0]

        # Hitung Similarity Score (Cosine via Dot Product pada vektor tersimpan)
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        # Top-K Ranking
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]

        return [(candidates[i], float(scores[i])) for i in top_idx]


# 5. Demo Eksekusi Skrip
if __name__ == "__main__":
    print("Membuat indeks dokumen...")
    
    docs = [
        FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
        FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
        FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
        FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
        FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
    ]

    fstore = FilteredVectorStore()
    fstore.add(docs)

    # Pencarian 1: Tanpa Filter (Semua dokumen)
    print("\n=== Hasil Pencarian: Semua Dokumen ===")
    results_all = fstore.search("which model is good at coding?", k=3)
    for doc, score in results_all:
        print(f"[{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")

    # Pencarian 2: Dengan Metadata Filter (Khusus 'anthropic')
    print("\n=== Hasil Pencarian: Khusus Katagori Anthropic ===")
    results_anthropic = fstore.search(
        "which model is good at coding?",
        k=3,
        filter_fn=lambda d: d.metadata.get("category") == "anthropic"
    )
    for doc, score in results_anthropic:
        print(f"[{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")