import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 1. Load Environment Variables dari file .env di root project
load_dotenv()

# Mengambil API Key dari .env (sesuai setup Phase sebelumnya)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY atau OPENAI_API_KEY belum diatur di file .env!")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY belum diatur di file .env! Silakan tambahkan ke .env terlebih dahulu.")

# 2. Inisialisasi Client OpenRouter (menggunakan interface OpenAI SDK)
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Helper function untuk membuat embedding via OpenRouter / OpenAI
def embed(texts: list[str]) -> list[list[float]]:
    """Generates vector embeddings for a list of texts using text-embedding-3-small."""
    response = llm_client.embeddings.create(
        model="text-embedding-3-small", 
        input=texts
    )
    # Urutkan berdasarkan index agar sesuai dengan urutan input
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

# 3. Connect ke Pinecone Vector Database
pc = Pinecone(api_key=pinecone_api_key)
INDEX_NAME = "ai-knowledge-base"

# 4. Buat Index jika belum ada (Serverless Indexing)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Dimensi bawaan text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created successfully!")

index = pc.Index(INDEX_NAME)

# 5. Data Dummy & Upsert ke Pinecone
documents = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases at inference time."},
    {"id": "d2", "text": "Fine-tuning updates model weights on task-specific training data."},
    {"id": "d3", "text": "ChromaDB and Pinecone are vector databases used for similarity search."}
]

print("\n--- Generating Embeddings & Upserting to Pinecone ---")
vectors = [
    {
        "id": doc["id"],
        "values": embed([doc["text"]])[0],
        "metadata": {"text": doc["text"], "source": "guide"}
    }
    for doc in documents
]

# Push/Upsert data ke Pinecone
index.upsert(vectors=vectors)
print("Upsert completed!")
print("Index Stats:", index.describe_index_stats())

# 6. Melakukan Query Similarity Search
query_text = "how to add external knowledge to an LLM without retraining?"
print(f"\n--- Querying Pinecone for: '{query_text}' ---")

q_vector = embed([query_text])[0]

# Query vector biasa
results = index.query(
    vector=q_vector,
    top_k=2,
    include_metadata=True
)

print("\nTop Match Results:")
for match in results["matches"]:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")

# 7. Melakukan Query dengan Metadata Filter
print("\n--- Querying with Metadata Filter (source == 'guide') ---")
filtered_results = index.query(
    vector=q_vector,
    top_k=2,
    filter={"source": {"$eq": "guide"}},
    include_metadata=True
)

for match in filtered_results["matches"]:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")