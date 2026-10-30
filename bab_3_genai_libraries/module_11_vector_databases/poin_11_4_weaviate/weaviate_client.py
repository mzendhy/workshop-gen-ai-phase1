import os
import weaviate
import weaviate.classes as wvc
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] API Key tidak ditemukan di berkas .env")
    exit()

# Menghubungkan ke OpenRouter API
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def get_embedding(text: str) -> list[float]:
    """Helper untuk membuat embedding menggunakan model OpenAI via OpenRouter."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

print("=== Module 11: Weaviate Vector Database Pattern ===")

# 1. Menghubungkan ke Weaviate Lokal (Docker Port 8081)
print("\n[1] Menghubungkan ke Docker Weaviate (localhost:8081)...")
client = weaviate.connect_to_local(port=8081)

try:
    # 2. Buat Collection
    collection_name = "Document"
    
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    documents = client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none()
    )
    print(f"[2] Collection '{collection_name}' berhasil dibuat.")

    # 3. Insert Data & Vector
    sample_text = "Pengembangan aplikasi web multimedia interaktif berbasis LLM."
    print(f"\n[3] Generating embedding untuk data: '{sample_text}'")
    vector_data = get_embedding(sample_text)

    doc_id = documents.data.insert(
        properties={"content": sample_text},
        vector=vector_data
    )
    print(f"    Data berhasil dimasukkan dengan ID: {doc_id}")

    # 4. Melakukan Vector Search (Near Vector)
    query_text = "Aplikasi web pembelajaran AI"
    print(f"\n[4] Melakukan Vector Search dengan Query: '{query_text}'")
    query_vector = get_embedding(query_text)

    response = documents.query.near_vector(
        near_vector=query_vector,
        limit=1
    )

    print("\n=== HASIL PENCARIAN WEAVIATE ===")
    for obj in response.objects:
        print(f"Content: {obj.properties['content']}")
        print(f"ID     : {obj.uuid}")

finally:
    # Tutup koneksi Weaviate
    client.close()
    print("\n[INFO] Koneksi Weaviate ditutup.")