import os
from openai import OpenAI
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path=env_path)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def chat():
    history = [
        {"role": "system", "content": "You are a helpful Python tutor."}
    ]
    
    sample_inputs = ["Halo, saya belajar Gen-AI", "Apa materi pertama saya?"]
    
    for user_input in sample_inputs:
        print(f"\nYou: {user_input}")
        history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-70b-instruct",
            messages=history
        )
        
        assistant_text = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_text})
        print(f"AI: {assistant_text}")

if __name__ == "__main__":
    chat()