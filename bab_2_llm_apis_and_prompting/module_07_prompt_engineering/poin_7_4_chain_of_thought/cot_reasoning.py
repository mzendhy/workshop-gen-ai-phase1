import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variable dari file .env
load_dotenv()

# Inisialisasi client OpenAI menggunakan OpenRouter API Key dan Base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Pilih model sesuai pilihanmu
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
# MODEL_NAME = "openai/gpt-4o-mini"

# Direct Prompt: Langsung meminta jawaban akhir
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, "
    "and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"
)

# CoT Prompt: Meminta model berpikir langkah demi langkah
COT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, "
    "and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?\n"
    "Think through this step by step before giving the final answer."
)

# Zero-shot CoT: Menambahkan instruksi eksplisit dan format keluaran
ZERO_SHOT_COT = (
    "Solve this problem. Think step by step, showing each calculation.\n"
    "Finally, state: ANSWER: $X.XXXXXX\n\n"
    "Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens "
    "and 400 output tokens. The model costs $3.00/M input and $15.00/M output.\n"
    "What is the daily cost?"
)

prompts = [
    ("Direct Prompt", DIRECT_PROMPT),
    ("CoT Prompt", COT_PROMPT),
    ("Zero-shot CoT Prompt", ZERO_SHOT_COT)
]

for label, prompt_text in prompts:
    print(f"=== {label} ===")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt_text}
        ]
    )
    
    output = response.choices[0].message.content
    print(output)
    print("\n" + "="*40 + "\n")