import os
from openai import OpenAI
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path=env_path)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

stream = client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
    stream=True
)

print("--- Streaming Response ---")
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()