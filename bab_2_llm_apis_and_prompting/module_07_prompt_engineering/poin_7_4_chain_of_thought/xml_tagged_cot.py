import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variable dari file .env
load_dotenv()

# Inisialisasi client OpenAI dengan OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
# MODEL_NAME = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

USER_PROMPT = (
    "A RAG pipeline retrieves 5 documents, each 400 tokens. "
    "The query is 50 tokens. The model has a 4096 token limit for context. "
    "How many tokens remain for the response?"
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    max_tokens=512,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
)

text_response = response.choices[0].message.content

# Mengambil teks di dalam tag <thinking> dan <answer> menggunakan Regex
thinking_match = re.search(r"<thinking>(.*?)</thinking>", text_response, re.DOTALL)
answer_match = re.search(r"<answer>(.*?)</answer>", text_response, re.DOTALL)

thinking_text = thinking_match.group(1).strip() if thinking_match else "Tag <thinking> tidak ditemukan"
answer_text = answer_match.group(1).strip() if answer_match else "Tag <answer> tidak ditemukan"

print("--- Hasil Response Mentah ---")
print(text_response)
print("\n--- Hasil Parsing ---")
print("Reasoning (Pemikiran):")
print(thinking_text)
print("\nAnswer (Jawaban Akhir):")
print(answer_text)