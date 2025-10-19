"""
upload_to_qdrant.py
Upload pre-embedded ToSDR documents into an embedded Qdrant instance (no Docker or Cloud needed).
"""

import json
from tqdm import tqdm
from qdrant_client import QdrantClient, models
import uuid

# Path to your local JSONL with embeddings
DATA_PATH = "data/processed/tosdr_docs_embedded.jsonl"
QDRANT_PATH = "data/qdrant_data"  # Folder where Qdrant stores its local DB
COLLECTION_NAME = "tosdr_docs"
EMBEDDING_SIZE = 1536  # 1536 for OpenAI's text-embedding-3-small


def make_uuid_from_str(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main():
    # 1️⃣ Start embedded Qdrant (runs inside Python, no Docker)
    print("🚀 Starting embedded Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)

    # 2️⃣ Create or recreate collection
    print("📁 Creating collection (if not exists)...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
        optimizers_config={"indexing_threshold": 20000},
    )

    # 3️⃣ Load documents and upload
    print(f"📤 Uploading embeddings from {DATA_PATH} ...")

    batch = []
    batch_size = 100  # adjust if needed
    count = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing documents"):
            doc = json.loads(line.strip())
            # Prepare a point for Qdrant
            point = models.PointStruct(
                id=make_uuid_from_str(doc["id"]),
                vector=doc["embedding"],
                payload={"source_id": doc["id"],
                         "source": doc["source"], 
                         "content": doc["content"]},
            )
            batch.append(point)
            # Batch insert for efficiency
            if len(batch) >= batch_size:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                count += len(batch)
                batch.clear()

        # Upload remaining points
        if batch:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            count += len(batch)

    print(f"✅ Successfully uploaded {count} documents to Qdrant collection '{COLLECTION_NAME}'!")
    client.close()

if __name__ == "__main__":
    main()
