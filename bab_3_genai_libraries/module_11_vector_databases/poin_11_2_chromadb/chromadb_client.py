import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variable dari .env di root
load_dotenv()

# 2. Inisialisasi OpenRouter Client (Meneruskan konfigurasi dari Phase sebelumnya)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Helper function untuk embedding menggunakan OpenRouter/OpenAI API
def get_embedding(texts: list[str]) -> list[list[float]]:
    # Menggunakan model embedding atau text model sesuai endpoint
    response = client_ai.embeddings.create(
        model="text-embedding-3-small", 
        input=texts
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

# 3. Persistent Client ChromaDB (Menyimpan data ke disk)
db_path = os.path.join(os.path.dirname(__file__), "chroma_db_data")
chroma_client = chromadb.PersistentClient(path=db_path)

collection = chroma_client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# 4. Tambah dokumen dengan pre-computed embeddings
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools."
]

ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"category": "technique", "index": i} for i in range(len(documents))]

print("Generating embeddings via OpenRouter...")
embeddings = get_embedding(documents)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Berhasil menyimpan data! Jumlah dokumen di collection: {collection.count()}")

# 5. Query Pencarian
query_text = "how do I add knowledge to an LLM without retraining?"
q_embed = get_embedding([query_text])[0]

results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

print("\nHasil Pencarian Teratas:")
for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
    print(f"- [{dist:.4f}] {doc} (Category: {meta['category']})")