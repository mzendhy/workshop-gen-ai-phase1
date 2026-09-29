import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables dari file .env di root project
load_dotenv()

# 2. Inisialisasi OpenAI Client yang mengarah ke OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# 3. Definisi Dataclass untuk Test Case Evaluasi
@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]  # Minimal salah satu keyword harus muncul di respon
    must_be_json: bool = False

# 4. Fungsi Utama Evaluasi Prompt (Menggunakan OpenRouter API)
def evaluate_prompt(system: str, cases: list[EvalCase], model_name: str = "meta-llama/llama-3.1-70b-instruct") -> dict:
    """Jalankan prompt ke test cases dan kembalikan pass rate serta detailnya."""
    results = []
    
    for case in cases:
        # Pemanggilan OpenRouter via OpenAI SDK
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.input_text},
            ],
            max_tokens=256,
            temperature=0.0,  # Gunakan temperature rendah agar hasil evaluasi konsisten
        )
        
        text = response.choices[0].message.content.strip()
        
        # Check 1: Pengecekan kemunculan keyword
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)
        
        # Check 2: Validasi format JSON (jika diminta)
        json_valid = True
        if case.must_be_json:
            try:
                # Bersihkan markdown fence jika model merespon dengan format ```json ... ```
                clean_text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                json.loads(clean_text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid
        
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    pass_rate = sum(r["passed"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "results": results}

# 5. Eksekusi Evaluasi Prompt
if __name__ == "__main__":
    print("--- EVALUASI PROMPT SYSTEM VIA OPENROUTER ---")

    # Prompt System yang ingin diuji kualitasnya
    CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

    # Dataset Test Cases
    test_cases = [
        EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
        EvalCase("Write a product description for headphones.", ["GENERATION"]),
        EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
        EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
        EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
    ]

    # Pilih model OpenRouter (bisa diganti ke "openai/gpt-4o-mini" jika mau)
    SELECTED_MODEL = "meta-llama/llama-3.1-70b-instruct"
    
    print(f"Menguji prompt menggunakan model: {SELECTED_MODEL}\n")
    
    # Jalankan evaluasi
    report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases, model_name=SELECTED_MODEL)

    # Cetak hasil Laporan
    print(f"Pass Rate: {report['pass_rate']:.0%}\n")
    print("Detail Hasil Evaluasi:")
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] '{r['input']}' -> '{r['response_preview']}'")