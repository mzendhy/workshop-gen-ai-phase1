import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load variabel dari file .env
load_dotenv()

# Mengambil API key OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API Key OpenRouter tidak ditemukan di .env!")

# 2. Inisialisasi OpenAI Client dengan Base URL milik OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

def embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Menghasilkan embedding untuk daftar teks menggunakan API OpenRouter.
    Menghasilkan numpy array berbentuk (jumlah_teks, dimensi_embedding).
    """
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    # Memastikan urutan vektor sesuai dengan urutan input teks
    vectors = sorted(response.data, key=lambda e: e.index)
    return np.array([v.embedding for v in vectors], dtype=np.float32)

if __name__ == "__main__":
    # Sampel teks untuk diuji
    sample_texts = [
        "Retrieval-Augmented Generation menggabungkan pencarian dengan LLM.",
        "RAG mengambil dokumen lalu menghasilkan jawaban berdasarkan dokumen tersebut.",
        "Menara Eiffel berada di kota Paris.",
        "Python adalah bahasa pemrograman yang sangat populer.",
        "Fine-tuning melatih ulang model menggunakan data baru."
    ]

    print("Sedang membuat embeddings via OpenRouter...")
    
    # Catatan mengenai pemilihan model:
    # 1. Untuk task Embeddings, gunakan model embedding seperti "text-embedding-3-small" atau "openai/text-embedding-3-small".
    # 2. Model seperti "meta-llama/llama-3.1-70b-instruct" atau "openai/gpt-4o-mini" adalah LLM Text Generation, 
    #    bukan model khusus pembuat Vektor Embedding.
    
    embedding_model = "openai/text-embedding-3-small"
    
    try:
        embeddings = embed(sample_texts, model=embedding_model)
        
        print("\n--- Hasil Generation Embeddings ---")
        print(f"Shape array : {embeddings.shape}")  # Contoh: (5, 1536)
        print(f"Panjang vektor ke-1 : {len(embeddings[0])}")
        print(f"Norm (L2) vektor ke-1 : {np.linalg.norm(embeddings[0]):.4f}")
        print("Contoh 5 nilai pertama dari vektor ke-1:", embeddings[0][:5])
        
    except Exception as e:
        print(f"\n[Error] Gagal membuat embedding: {e}")