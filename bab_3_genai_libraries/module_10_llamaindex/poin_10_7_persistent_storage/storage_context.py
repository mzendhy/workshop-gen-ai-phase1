import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext, 
    load_index_from_storage, 
    Settings
)
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load Environment Variables (.env)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 2. Konfigurasi Settings LlamaIndex menggunakan OpenRouter
Settings.llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.1-70b-instruct",
    max_tokens=1024,
)

# Fix Error 401: Arahkan embedding ke endpoint OpenRouter
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
)

# Path penyimpanan lokal
PERSIST_DIR = "output/llama_index_store"

# 3. Membuat Dokumen & Membangun Index
documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

index = VectorStoreIndex.from_documents(documents)

# Simpan index ke penyimpanan lokal (Disk)
index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"Index berhasil disimpan ke: {PERSIST_DIR}")

# 4. Memuat kembali Index dari Penyimpanan (Persisted Storage)
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
loaded_index = load_index_from_storage(storage_context)

# 5. Uji Coba Query Engine
query_engine = loaded_index.as_query_engine()
response = query_engine.query("How does LlamaIndex handle persistence?")

print("\nHasil Query setelah Reload Index:")
print(response)