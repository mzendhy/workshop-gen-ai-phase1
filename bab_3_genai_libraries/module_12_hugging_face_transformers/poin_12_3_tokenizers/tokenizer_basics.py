from transformers import AutoTokenizer

def main():
    # 1. Load tokenizer lokal dari Hugging Face Hub (GPT-2)
    model_name = "gpt2"
    print(f"--- Loading Tokenizer: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # PERBAIKAN: Set pad_token ke eos_token untuk mendukung batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

    # 2. Encode: Mengubah teks menjadi Token IDs
    tokens = tokenizer.encode(text)
    print(f"\nOriginal Text: {text}")
    print(f"Token Count: {len(tokens)}")
    print(f"Token IDs: {tokens}")

    # 3. Decode: Mengubah Token IDs kembali menjadi teks
    decoded = tokenizer.decode(tokens)
    print(f"Decoded Text: {decoded}")

    # 4. Inspect Individual Tokens
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Individual Tokens: {token_strings}")

    # 5. Batch Encoding dengan Padding & Truncation
    batch_sentences = [
        "What is RAG?",
        "Explain vector databases in detail with examples."
    ]
    encoded_batch = tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"  # PyTorch tensors
    )

    print("\n--- Batch Encoding Results ---")
    print(f"Batch input_ids shape: {encoded_batch['input_ids'].shape}")
    print(f"Attention mask shape: {encoded_batch['attention_mask'].shape}")

    # 6. Utility Function: Menghitung Token lokal
    prompt = "Summarise the following document: " + "word " * 500
    print(f"\nCalculated Prompt Token Count: {len(tokenizer.encode(prompt))}")

if __name__ == "__main__":
    main()