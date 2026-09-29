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

stream = client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain Python lists in 3 short bullet points."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # Baris baru setelah streaming selesai