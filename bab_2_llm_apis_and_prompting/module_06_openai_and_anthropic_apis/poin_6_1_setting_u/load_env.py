import os
from dotenv import load_dotenv

# load_dotenv() akan mencari .env di folder tempat script dijalankan atau folder di atasnya
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))

anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "Key tidak ditemukan")
openai_key = os.environ.get("OPENAI_API_KEY", "Key tidak ditemukan")

print("--- Check Environment Variables ---")
print(f"Anthropic Key Loaded: {anthropic_key[:10]}...")
print(f"OpenAI Key Loaded   : {openai_key[:10]}...")