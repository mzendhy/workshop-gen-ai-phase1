import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables dari file .env di root
load_dotenv()

# 2. Inisialisasi OpenAI Client yang diarahkan ke OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY"),
)

# 3. Model OpenRouter (Llama 3.1 70B Instruct atau GPT-4o Mini)
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
# MODEL_NAME = "openai/gpt-4o-mini"

# ---------------------------------------------------------
# Few-Shot System Prompt
# Memberikan instruksi + contoh konkret (Input -> Output)
# ---------------------------------------------------------
FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string, extract: model name, task, and score as a JSON object.

Examples:
Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation."""

# Data uji (Test Inputs)
test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]


def run_fewshot_extraction():
    print("=================== FEW-SHOT EXTRACTION RESULT ===================\n")

    for text in test_inputs:
        # Susun messages dengan format OpenAI SDK (system prompt & user content)
        messages = [
            {"role": "system", "content": FEW_SHOT_SYSTEM},
            {"role": "user", "content": text},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=128,
            temperature=0.0,  # Gunakan 0.0 agar output konsisten/deterministik
            messages=messages,
        )

        output_text = response.choices[0].message.content.strip()

        print(f"Input : {text}")
        print(f"Output: {output_text}\n" + "-" * 50)


if __name__ == "__main__":
    run_fewshot_extraction()