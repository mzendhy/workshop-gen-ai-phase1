import os
import re
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables (.env)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

# Inisialisasi OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# 2. Structure Data Chunk
@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

# 3. Fungsi Chunking Berdasarkan Kalimat & Overlap
def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 300,
    overlap_chars: int = 50,
) -> list[Chunk]:
    """
    Memotong teks menjadi chunk yang menghormati batas kalimat 
    dan menambahkan overlap agar kontekstual tidak hilang.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[Chunk] = []
    current = ""
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            end = char_offset + len(current)
            chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))
            chunk_idx += 1
            
            overlap_start = max(0, len(current) - overlap_chars)
            overlap_text = current[overlap_start:]
            current = (overlap_text + " " + sentence).strip()
            char_offset = end - len(overlap_text)
        else:
            current = candidate

    if current.strip():
        end = char_offset + len(current)
        chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))

    return chunks

# 4. Main Execution Step
if __name__ == "__main__":
    sample_document = """
    Large language models (LLMs) are neural networks trained on vast amounts of text data.
    They learn to predict the next token in a sequence, which gives them broad language understanding.
    Models like GPT-4 and Claude are examples of LLMs used in production today.
    Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
    Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
    This allows the model to answer questions about recent events or private data it was never trained on.
    The retrieval step in RAG typically uses embedding-based semantic search.
    A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
    These documents are then injected into the LLM's context window alongside the query.
    """

    print("--- 1. Memproses Text Chunking ---")
    # Menggunakan 'text' sebagai keyword argument (sesuai parameter fungsi)
    chunks = chunk_by_sentences(
        text=sample_document.strip(), 
        doc_id="doc_intro_llm", 
        max_chars=250, 
        overlap_chars=40
    )

    for c in chunks:
        print(f"[Chunk {c.chunk_index}] ({c.char_start}-{c.char_end} char): {c.text}\n")

    print(f"Total Chunk Dihasilkan: {len(chunks)}")

    print("\n--- 2. Mengirimkan Chunk ke LLM OpenRouter ---")
    MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
    
    first_chunk_text = chunks[0].text
    prompt = f"Rangkum chunk teks berikut dalam 1 kalimat singkat:\n\n{first_chunk_text}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print(f"Model ({MODEL_NAME}) Response:")
    print(response.choices[0].message.content)