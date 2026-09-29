import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables dari file .env
load_dotenv()

# 2. Inisialisasi OpenAI Client dengan endpoint OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY"),
)

# 3. Pilih Model OpenRouter
# Kamu bisa pakai model Llama 3.1 70B Instruct atau GPT-4o Mini
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
# MODEL_NAME = "openai/gpt-4o-mini"

# ---------------------------------------------------------
# Definisi System Prompts (Perbandingan Weak vs Strong)
# ---------------------------------------------------------

# Weak System Prompt (Terlalu samar, tanpa aturan/format eksplisit)
WEAK_SYSTEM = "You are an AI assistant."

# Strong System Prompt (Peran jelas, aturan spesifik, dan instruksi format output)
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems.
- Suggest concrete improvements with code examples.
- Explain WHY each issue matters.

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue -> Impact -> Fix."""

# Kode Python sampel yang akan direview
CODE_TO_REVIEW = """Review this function:
def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""


def test_system_prompt(prompt_label: str, system_instruction: str):
    print(f"\n=================== {prompt_label} ===================")

    # Susun payload message untuk OpenAI/OpenRouter SDK
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": CODE_TO_REVIEW},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, max_tokens=1024, temperature=0.3, messages=messages
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    # Jalankan pengujian
    test_system_prompt("WEAK SYSTEM PROMPT RESULT", WEAK_SYSTEM)
    test_system_prompt("STRONG SYSTEM PROMPT RESULT", STRONG_SYSTEM)