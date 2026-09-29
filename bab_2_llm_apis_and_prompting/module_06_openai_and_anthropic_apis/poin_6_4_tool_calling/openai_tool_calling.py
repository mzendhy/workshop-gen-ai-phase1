import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Otomatis melacak dan memuat file .env di folder root proyek
load_dotenv(find_dotenv())

# Inisialisasi client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# 1. Definisi skema Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"]
            }
        }
    }
]

# 2. Fungsi mock lokal
def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})

# 3. Prompt user
messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

# Menggunakan model openai/gpt-4o-mini
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    tools=tools,
    messages=messages
)

response_message = response.choices[0].message

# 4. Eksekusi Tool Call jika diminta oleh model
if response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = get_model_info(**args)
    
    print(f"Tool called: {tool_call.function.name} ({args})")
    print(f"Tool result: {result}")
    
    messages.append(response_message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    final = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages
    )
    print("\nFinal answer:")
    print(final.choices[0].message.content)
else:
    print(response_message.content)