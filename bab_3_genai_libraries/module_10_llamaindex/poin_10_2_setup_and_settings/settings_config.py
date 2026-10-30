import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load environment variables dari file .env
load_dotenv()

# Ambil API Key OpenRouter (fallback ke OPENAI_API_KEY jika kamu menamai variabelnya begitu)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openrouter_api_key:
    raise ValueError("API Key tidak ditemukan! Pastikan OPENROUTER_API_KEY sudah diatur di file .env")

# 2. Konfigurasi Global LLM LlamaIndex via OpenRouter
Settings.llm = OpenAI(
    model="meta-llama/llama-3.1-70b-instruct",  # Atau pakai "openai/gpt-4o-mini"
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",    # Base URL OpenRouter
    temperature=0.1,
    max_tokens=1024,
)

# 3. Konfigurasi Global Embedding Model
# Catatan: Untuk embedding via OpenRouter, pastikan endpoint/key mendukungnya, 
# atau gunakan OpenAI API Key asli / model embedding bawaan LlamaIndex jika diperlukan.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
)

# 4. Pengaturan ukuran Chunk Global
Settings.chunk_size = 512       # Jumlah token per chunk
Settings.chunk_overlap = 64     # Overlap token antar chunk

if __name__ == "__main__":
    print("=== Konfigurasi Global LlamaIndex Berhasil Diterapkan ===")
    print(f"LLM Model      : {Settings.llm.model}")
    print(f"Embedding Model: {Settings.embed_model.model_name}")
    print(f"Chunk Size     : {Settings.chunk_size}")
    print(f"Chunk Overlap  : {Settings.chunk_overlap}")