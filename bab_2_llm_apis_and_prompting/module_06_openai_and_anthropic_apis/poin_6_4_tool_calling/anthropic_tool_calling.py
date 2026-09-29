import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Otomatis mencari file .env di folder root
load_dotenv(find_dotenv())

# Inisialisasi client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# 1. Fungsi mock lokal
def get_model_info(model_name: str) -> dict:
    db = {
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o": {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro": {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name, {"error": f"Unknown model: {model_name}"})

# 2. Schema Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window size and cost per 1K tokens for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model identifier, e.g. 'gpt-4o' or 'claude-sonnet-4-5'."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]

# 3. Prompt user
messages = [
    {"role": "user", "content": "How large is the context window of claude-sonnet-4-5?"}
]

# Menggunakan model gpt-4o-mini yang sangat stabil untuk tool calling via OpenRouter
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    tools=tools,
)

response_message = response.choices[0].message

# 4. Penanganan Tool Calling
if response_message.tool_calls:
    messages.append(response_message)
    
    for tool_call in response_message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_model_info(**args)
        
        print(f"Tool called: {tool_call.function.name} ({args})")
        print(f"Tool result: {result}")
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Kirim balik hasil fungsi ke LLM untuk mendapatkan respons akhir
    final = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages
    )
    print("\nFinal answer:")
    print(final.choices[0].message.content)
else:
    print(response_message.content)