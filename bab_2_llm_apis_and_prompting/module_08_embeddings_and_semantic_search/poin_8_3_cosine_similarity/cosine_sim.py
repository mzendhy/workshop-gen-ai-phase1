import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables dari .env
load_dotenv()

# 2. Inisialisasi client OpenAI yang diarahkan ke OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
)

# Gunakan model embedding (Catatan: untuk OpenRouter/OpenAI gunakan model embedding yang valid)
MODEL_NAME = "text-embedding-3-small"

# Fungsi untuk menghitung Cosine Similarity antara dua vektor 1-D
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# Fungsi untuk menghitung Pairwise Cosine Similarity untuk seluruh matriks
def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms
    return (normed @ normed.T).astype(np.float32)

# Fungsi untuk mengambil embedding asli dari API OpenRouter
def get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=MODEL_NAME,
        input=texts
    )
    # Urutkan berdasarkan indeks agar sesuai urutan input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    embeddings = [item.embedding for item in sorted_data]
    return np.array(embeddings, dtype=np.float32)

if __name__ == "__main__":
    # Teks sampel untuk diuji kemiripannya
    texts = [
        "Apa itu Retrieval-Augmented Generation (RAG)?",
        "Jelaskan konsep RAG pada LLM.",
        "Resep membuat nasi goreng enak.",
        "Bagaimana cara memasak nasi goreng di rumah?"
    ]

    print(" Mengambil embedding dari OpenRouter API...")
    embeddings_matrix = get_embeddings(texts)
    print(f" Bentuk matriks embedding (Shape): {embeddings_matrix.shape}\n")

    # Hitung matriks kesamaan berpasangan
    sim_matrix = pairwise_similarity(embeddings_matrix)

    print(" Matrix Cosine Similarity Berpasangan:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            print(f"• Teks [{i}] vs Teks [{j}]: {sim_matrix[i, j]:.4f}")
            print(f"   - Teks {i}: \"{texts[i]}\"")
            print(f"   - Teks {j}: \"{texts[j]}\"")
            print("-" * 50)