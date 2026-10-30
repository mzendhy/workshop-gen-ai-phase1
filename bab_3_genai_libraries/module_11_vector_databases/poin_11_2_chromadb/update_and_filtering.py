import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client_ai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_embedding(texts: list[str]) -> list[list[float]]:
    response = client_ai.embeddings.create(
        model="text-embedding-3-small", 
        input=texts
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

# Membuka DB yang sama dengan skrip pertama
db_path = os.path.join(os.path.dirname(__file__), "chroma_db_data")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="ai_knowledge_base")

# 1. Metadata Filtering (Pre-filtering berdasarkan tag metadata)
print("=== METADATA FILTERING ===")
query_text = "What is RAG?"
q_embed = get_embedding([query_text])[0]

filtered_results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}},  # Filter metadata
    include=["documents", "distances"]
)

for doc, dist in zip(filtered_results["documents"][0], filtered_results["distances"][0]):
    print(f"- [{dist:.4f}] {doc}")

# 2. Update Dokumen
print("\n=== UPDATE DOKUMEN ===")
updated_doc = "RAG retrieves external documents and injects them into the LLM context."
collection.update(
    ids=["doc_0"],
    documents=[updated_doc],
    embeddings=get_embedding([updated_doc])
)
print("Dokumen 'doc_0' berhasil diperbarui.")

# 3. Delete Dokumen
print("\n=== DELETE DOKUMEN ===")
print(f"Jumlah awal: {collection.count()}")
collection.delete(ids=["doc_4"])
print(f"Setelah menghapus 'doc_4', jumlah dokumen: {collection.count()}")