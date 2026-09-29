import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Pastikan URL menyertakan 'https://' secara eksplisit
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = "openai/gpt-4o-mini"

def extract_entities(text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract entities. Return JSON with this exact schema:\n"
                    '{"people": [string], "organizations": [string], "locations": [string]}'
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.1
    )
    
    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":
    sample_text = "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
    result = extract_entities(sample_text)
    print("Hasil JSON Mode:")
    print(json.dumps(result, indent=2))