import os
import json

RAW_DIR = "data/raw/text"
OUTPUT_FILE = "data/processed/tosdr_docs.jsonl"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def process_text_files():
    docs = []
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append({
                "id": filename.replace(".txt", ""),
                "source": filename,
                "content": text
            })
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc) + "\n")

    print(f"✅ Processed {len(docs)} files → {OUTPUT_FILE}")

if __name__ == "__main__":
    process_text_files()
