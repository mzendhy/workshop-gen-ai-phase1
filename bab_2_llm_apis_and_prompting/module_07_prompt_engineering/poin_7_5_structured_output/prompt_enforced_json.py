import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gunakan OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Menggunakan model Meta-Llama 3.1 70B Instruct melalui OpenRouter
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"

SYSTEM_PROMPT = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment."
]

def extract_company_info(text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.1
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Pembersihan tag markdown jika model secara tidak sengaja menambahkannya (seperti ```json ... ```)
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
        
    return json.loads(raw)

if __name__ == "__main__":
    for text in texts:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print("-" * 40)