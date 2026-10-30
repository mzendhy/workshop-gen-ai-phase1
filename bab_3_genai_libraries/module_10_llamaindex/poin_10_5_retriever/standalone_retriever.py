import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load environment variables dari file .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env!")

# 2. Konfigurasi Global Settings LlamaIndex menggunakan OpenRouter
# Menggunakan model Meta-Llama / GPT-4o-Mini via OpenRouter
Settings.llm = OpenAILike(
    model="meta-llama/llama-3.1-70b-instruct",  # atau "openai/gpt-4o-mini"
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    temperature=0.1,
    max_tokens=1024,
)

# Untuk embedding, kita gunakan konfigurasi embedding OpenRouter / OpenAI
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

# 3. Buat Dokumen Dummy (Sesuai Materi Modul 10.5 PDF)
documents = [
    Document(
        text="ChromaDB is an open-source embedding database with a simple Python API.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="Pinecone is a managed vector database designed for production workloads.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="FAISS is a library by Meta for efficient similarity search on dense vectors.",
        metadata={"tag": "vector-search"}
    ),
    Document(
        text="pgvector adds vector similarity search to PostgreSQL.",
        metadata={"tag": "vector-db"}
    ),
    Document(
        text="Weaviate is an open-source vector database with a GraphQL API.",
        metadata={"tag": "vector-db"}
    ),
]

# 4. Buat VectorStoreIndex dari dokumen
index = VectorStoreIndex.from_documents(documents)

# 5. Inisialisasi Standalone VectorIndexRetriever
# Retriever ini digunakan secara independen tanpa query engine (hanya mengambil node/chunk)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# 6. Jalankan Retrieval tanpa melakukan sintesis jawaban LLM
nodes = retriever.retrieve("open source vector databases")

# 7. Tampilkan Hasil
print("=== Retrieved Nodes (Standalone Retriever) ===")
for node in nodes:
    print(f"[{node.score:.4f}] {node.text}")
    print(f"  tag: {node.metadata.get('tag')}\n")