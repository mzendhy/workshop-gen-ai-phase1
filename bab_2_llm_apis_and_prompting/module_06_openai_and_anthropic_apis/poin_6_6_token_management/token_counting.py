import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Menggunakan OpenRouter API Key dan Base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

# 1. Menghitung token secara lokal sebelum mengirim request (Estimasi via tiktoken)
def count_tokens_locally(text: str, model_encoding: str = "cl100k_base") -> int:
    """Menghitung jumlah token string teks menggunakan tiktoken secara lokal."""
    encoding = tiktoken.get_encoding(model_encoding)
    return len(encoding.encode(text))

prompt_text = "Explain the transformer architecture."
system_text = "You are a concise assistant."

estimated_input_tokens = count_tokens_locally(system_text + " " + prompt_text)
print(f"Estimated input tokens (Local): {estimated_input_tokens}")

# 2. Estimasi Biaya (PRICING per 1M tokens)
PRICING = {
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.35, "output": 0.40},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Menghitung estimasi biaya dalam USD per 1M token."""
    if model not in PRICING:
        # Nilai default jika model tidak ada di daftar pricing internal
        return 0.0
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# Contoh pemanggilan ke OpenRouter dengan model Meta Llama / OpenAI
model_name = "meta-llama/llama-3.1-70b-instruct"
# Jika ingin pakai GPT-4o-Mini:
# model_name = "openai/gpt-4o-mini"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text}
    ],
    max_tokens=150
)

# 3. Mengambil token acuan resmi dari respon API OpenRouter
if response.usage:
    actual_input = response.usage.prompt_tokens
    actual_output = response.usage.completion_tokens
    total_cost = estimate_cost(model_name, actual_input, actual_output)
    
    print(f"\n--- Output Respons API ---")
    print(f"Model Used: {model_name}")
    print(f"Actual Input Tokens: {actual_input}")
    print(f"Actual Output Tokens: {actual_output}")
    print(f"Estimated Cost: ${total_cost:.6f}")
    print(f"Response Text:\n{response.choices[0].message.content}")

# 4. Pengecekan Batas Context Window
CONTEXT_LIMITS = {
    "meta-llama/llama-3.1-70b-instruct": 128_000,
    "openai/gpt-4o-mini": 128_000,
}

def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit

print(f"\nFits in context? {fits_in_context(model_name, estimated_input_tokens)}")