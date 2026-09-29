import os
from openai import OpenAI
from dotenv import load_dotenv

# Path presisi ke file .env di root folder
env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path=env_path)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    model="meta-llama/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "What is retrieval-augmented generation?"}
    ]
)

print(response.choices[0].message.content)