import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI
from itertools import islice

# --- Load environment variables ---
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = "text-embedding-3-small"

# --- Azure OpenAI client ---
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# --- File paths ---
input_file = "data/processed/tosdr_docs_chunked.jsonl"
output_file = "data/processed/tosdr_docs_embedded.jsonl"

# --- Parameters ---
BATCH_SIZE = 5      # adjust based on chunk length
RETRY_LIMIT = 3      # retry failed calls
SLEEP_BETWEEN = 0.2  # pause between batches

# --- Helper: batch generator ---
def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

# --- Helper: read processed IDs from existing output file ---
def get_processed_ids(output_path):
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    processed.add(doc["id"])
                except json.JSONDecodeError:
                    continue
    return processed

# --- Embedding logic ---
def embed_documents_batched():
    # Load input data
    with open(input_file, "r", encoding="utf-8") as infile:
        all_lines = infile.readlines()
    
    print(f"📄 Total chunks in input: {len(all_lines)}")

    # Load processed IDs if resuming
    processed_ids = get_processed_ids(output_file)
    print(f"⏩ Already processed: {len(processed_ids)} chunks")

    # Filter unprocessed lines
    remaining_lines = [line for line in all_lines if json.loads(line)["id"] not in processed_ids]
    print(f"🚀 Remaining to embed: {len(remaining_lines)} chunks")

    if not remaining_lines:
        print("✅ All documents already processed!")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open file in append mode so we don’t overwrite progress
    with open(output_file, "a", encoding="utf-8") as outfile:
        for batch in tqdm(list(batched(remaining_lines, BATCH_SIZE)), desc="Embedding in batches"):
            docs = [json.loads(line) for line in batch]
            inputs = [doc["content"].strip() for doc in docs if doc["content"].strip()]

            if not inputs:
                continue

            # Retry logic for transient errors
            for attempt in range(RETRY_LIMIT):
                try:
                    response = client.embeddings.create(
                        model=EMBED_MODEL,
                        input=inputs
                    )
                    break
                except Exception as e:
                    print(f"⚠️ Error on attempt {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
            else:
                print("❌ Skipping batch after multiple failures")
                continue

            # Save embeddings incrementally
            embeddings = [r.embedding for r in response.data]
            for doc, emb in zip(docs, embeddings):
                embedded_doc = {
                    "id": doc["id"],
                    "source": doc["source"],
                    "content": doc["content"],
                    "embedding": emb
                }
                outfile.write(json.dumps(embedded_doc) + "\n")

            outfile.flush()  # flush every batch
            time.sleep(SLEEP_BETWEEN)

    print(f"\n✅ All embeddings saved to {output_file}")

def regenerate_missing_embeddings():
    # Re-generate only missing embeddings
    # === File paths ===
    input_file = "data/processed/tosdr_docs_chunked.jsonl"
    output_file = "data/processed/tosdr_docs_embedded.jsonl"

    # === Load existing embedded IDs (if any) ===
    existing_ids = set()
    if os.path.exists(output_file):
        print(f"Loading existing embeddings from {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue

    print(f"Found {len(existing_ids):,} already embedded documents.")

    # === Load all input docs ===
    all_docs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            all_docs.append(doc)

    print(f"Total input docs: {len(all_docs):,}")

    # === Find missing ones ===
    missing_docs = [doc for doc in all_docs if doc["id"] not in existing_ids]
    print(f"Missing embeddings: {len(missing_docs):,}")

    if not missing_docs:
        print("✅ All documents already embedded. Nothing to do!")
        exit()

    # === Generate embeddings for missing ones ===
    with open(output_file, "a", encoding="utf-8") as out_f:
        for doc in tqdm(missing_docs, desc="Embedding missing docs"):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=doc["content"]
                )
                embedding = response.data[0].embedding

                out_record = {
                    "id": doc["id"],
                    "source": doc["source"],
                    "content": doc["content"],
                    "embedding": embedding
                }

                out_f.write(json.dumps(out_record) + "\n")

            except Exception as e:
                print(f"⚠️ Error embedding {doc['id']}: {e}")
                continue

    print("✅ Missing embeddings successfully added to output file!")

if __name__ == "__main__":
    embed_documents_batched()
    regenerate_missing_embeddings()