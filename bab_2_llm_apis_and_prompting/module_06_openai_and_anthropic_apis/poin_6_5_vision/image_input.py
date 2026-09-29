import os
import base64
import urllib.request
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Otomatis melacak dan memuat file .env di folder root
load_dotenv(find_dotenv())

# Inisialisasi client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Fungsi untuk mengonversi bytes/file gambar ke Base64
def encode_image_from_url(url: str) -> str:
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response:
        image_data = response.read()
        return base64.b64encode(image_data).decode("utf-8")

# Fungsi untuk menganalisis gambar via Vision API
def analyze_image(image_url: str):
    print("--- Mengunduh dan Menganalisis Gambar ---")
    try:
        base64_image = encode_image_from_url(image_url)
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Jelaskan secara ringkas apa yang terlihat pada gambar ini dalam bahasa Indonesia."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        print("\nHasil Analisis Vision:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")

if __name__ == "__main__":
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    
    analyze_image(sample_url)