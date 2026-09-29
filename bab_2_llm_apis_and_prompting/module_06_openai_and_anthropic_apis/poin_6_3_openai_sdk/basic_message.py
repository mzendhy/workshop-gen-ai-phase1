import os
from openai import OpenAI
from dotenv import load_dotenv

# Path ke file .env di root project (naik 3 folder)
env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path=env_path)

# Inisialisasi OpenAI client dengan endpoint OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "You are a concise technical writer. Answer in plain English, no jargon."},
        {"role": "user", "content": "Explain what a vector database does."}
    ]
)

print(response.choices[0].message.content)