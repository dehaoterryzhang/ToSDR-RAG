import json
import os
from textwrap import wrap

INPUT_FILE = "data/processed/tosdr_docs.jsonl"
OUTPUT_FILE = "data/processed/tosdr_docs_chunked.jsonl"

# Define chunking parameters
WORDS_PER_CHUNK = 1000
OVERLAP = 100  # optional overlap for context continuity

def chunk_text(text, chunk_size=WORDS_PER_CHUNK, overlap=OVERLAP):
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap  # slide window with overlap
    return chunks

def chunk_documents():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    chunked_docs = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            doc = json.loads(line)
            content = doc["content"]

            # Only chunk if needed
            words = content.split()
            if len(words) > WORDS_PER_CHUNK:
                chunks = chunk_text(content)
                for i, chunk in enumerate(chunks):
                    chunked_doc = {
                        "id": f"{doc['id']}_chunk{i+1}",
                        "source": doc["source"],
                        "content": chunk
                    }
                    outfile.write(json.dumps(chunked_doc) + "\n")
                    chunked_docs.append(chunked_doc)
            else:
                outfile.write(json.dumps(doc) + "\n")
                chunked_docs.append(doc)

    print(f"✅ Chunked and saved {len(chunked_docs)} documents → {OUTPUT_FILE}")

if __name__ == "__main__":
    chunk_documents()
