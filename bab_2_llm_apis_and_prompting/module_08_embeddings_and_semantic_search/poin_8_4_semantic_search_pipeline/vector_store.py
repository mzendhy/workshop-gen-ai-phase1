import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables dari .env
load_dotenv()

# Gunakan OpenRouter API Key dan Base URL
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# 2. Structure Data
@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

# 3. Embedding Helper via OpenRouter
def embed_batch(texts: list[str], model: str = "openai/text-embedding-3-small") -> np.ndarray:
    """Mengambil embedding menggunakan OpenRouter API client."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

# 4. Vector Store Class (Semantic Search Engine)
class VectorStore:
    def __init__(self, embed_model: str = "openai/text-embedding-3-small"):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: list[Document]) -> None:
        """Meng-embed dan mengindeks daftar dokumen."""
        texts = [d.text for d in documents]
        vectors = embed_batch(texts, model=self.embed_model)

        # Normalisasi vector untuk Cosine Similarity cepat via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)
        print(f"Indeks sekarang berisi {len(self._documents)} dokumen.")

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Mencari k dokumen paling relevan untuk suatu query."""
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("Belum ada dokumen yang diindeks.")

        # Embed dan normalisasi query
        q_vec = embed_batch([query], model=self.embed_model)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_normed = (q_vec / q_norm).astype(np.float32)

        # Cosine Similarity via Dot Product (matrix @ vector)
        scores = self._matrix @ q_normed

        # Ambil Top-k indeks terbesar
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

# 5. Demo Pengujian Pipeline RAG & Semantic Search
if __name__ == "__main__":
    CORPUS = [
        Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation."),
        Document("d02", "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search."),
        Document("d03", "Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a dataset."),
        Document("d04", "Prompt engineering involves designing and optimising input prompts to guide language models."),
        Document("d05", "LangChain is a Python framework for building applications with large language models.")
    ]

    print("--- 1. Mengindeks Dokumen ---")
    store = VectorStore()
    store.add_documents(CORPUS)

    print("\n--- 2. Pengujian Semantic Search ---")
    query = "How does RAG work?"
    print(f"Query: '{query}'\n")

    results = store.search(query, k=3)
    for r in results:
        print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.document.id} | Text: {r.document.text}")

    print("\n--- 3. Pengujian Prompting / Generasi dengan OpenRouter ---")
    # Contoh pemanggilan LLM dengan model meta-llama/llama-3.1-70b-instruct
    context = "\n".join([r.document.text for r in results])
    prompt = f"Gunakan konteks berikut untuk menjawab pertanyaan.\nKonteks:\n{context}\n\nPertanyaan: {query}"

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",  # atau "openai/gpt-4o-mini"
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\nHasil Jawaban LLM:")
    print(response.choices[0].message.content)