import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# ==============================
# CONFIG
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDRANT_PATH = os.path.join(PROJECT_ROOT, "data/qdrant_data")
EVAL_FILE = os.path.join(PROJECT_ROOT, "eval/retrieval_eval_ground_truth.json")
COLLECTION_NAME = "tosdr_docs"
TOP_K = 5
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = "text-embedding-3-small"

# ==============================
# CONNECT TO EMBEDDED QDRANT
# ==============================
client = QdrantClient(path=QDRANT_PATH)
print(f"✅ Connected to embedded Qdrant at {QDRANT_PATH}")


# ==============================
# CONNECT TO AZURE OPENAI
# ==============================
client_openai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# ==============================
# LOAD EVAL DATA
# ==============================
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"📄 Loaded {len(eval_data)} evaluation queries from {EVAL_FILE}")

# ==============================
# EVALUATION HELPERS
# ==============================
def embed_query(query_text: str) -> list[float]:
    """Generate query embedding via Azure OpenAI."""
    response = client_openai.embeddings.create(
        model=EMBED_MODEL,
        input=query_text
    )
    return response.data[0].embedding


def compute_hit_rate(results, ground_truths, k=TOP_K):
    """Compute Hit Rate@k given retrieved results and ground truth IDs."""
    hits = 0
    for gt, res in zip(ground_truths, results):
        topk_ids = [r.payload.get("source_id") for r in res[:k]]
        if gt in topk_ids:
            hits += 1
    return hits / len(ground_truths)


def run_vector_search(query_text):
    """Run pure vector search using query embedding."""
    query_vector = embed_query(query_text)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
    )
    return results

def run_hybrid_search(query_text):
    """Run hybrid (text + vector) search using RRF (Reciprocal Rank Fusion)."""
    # Get vector search results
    query_vector = embed_query(query_text)
    vector_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K * 3,  # Get more results for fusion
    )
    
    # Enhanced text-based search with multiple strategies
    query_keywords = query_text.lower().split()
    
    # Strategy 1: Individual keywords (broader match)
    keyword_filter = rest.Filter(
        should=[
            rest.FieldCondition(
                key="content",
                match=rest.MatchText(text=keyword)
            ) for keyword in query_keywords if len(keyword) > 2  # Skip very short words
        ]
    )
    
    # Strategy 2: Phrase search (exact phrase matching)
    phrase_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="content",
                match=rest.MatchText(text=query_text.lower())
            )
        ]
    )
    
    # Get results from both text search strategies
    text_results_broad = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=keyword_filter,
        limit=TOP_K * 2,
        with_payload=True,
        with_vectors=False
    )[0] if len(query_keywords) > 0 else []
    
    text_results_phrase = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=phrase_filter,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False
    )[0]
    
    # Combine and deduplicate text results
    text_results_dict = {}
    
    # Add phrase matches first (higher priority)
    for point in text_results_phrase:
        point_id = point.payload.get("source_id", str(point.id))
        text_results_dict[point_id] = {'point': point, 'type': 'phrase'}
    
    # Add keyword matches
    for point in text_results_broad:
        point_id = point.payload.get("source_id", str(point.id))
        if point_id not in text_results_dict:
            text_results_dict[point_id] = {'point': point, 'type': 'keyword'}
    
    # Implement enhanced RRF with different weights
    def rrf_score(rank, k=60):
        """Calculate RRF score: 1 / (k + rank)"""
        return 1.0 / (k + rank)
    
    # Create score dictionaries with different weights
    vector_scores = {}
    text_scores = {}
    
    # Score vector results (standard weight)
    for i, point in enumerate(vector_results):
        point_id = point.payload.get("source_id", str(point.id))
        vector_scores[point_id] = rrf_score(i + 1, k=60)
    
    # Score text results with different weights based on match type
    text_items = list(text_results_dict.items())
    for i, (point_id, item) in enumerate(text_items):
        base_score = rrf_score(i + 1, k=30)  # Lower k for text = higher scores
        # Boost phrase matches
        if item['type'] == 'phrase':
            text_scores[point_id] = base_score * 1.5  # 50% boost for exact phrases
        else:
            text_scores[point_id] = base_score
    
    # Combine scores using weighted RRF
    all_point_ids = set(vector_scores.keys()) | set(text_scores.keys())
    combined_scores = {}
    
    for point_id in all_point_ids:
        # Weight: 0.7 for vector, 0.3 for text (adjust as needed)
        vector_component = vector_scores.get(point_id, 0) * 0.7
        text_component = text_scores.get(point_id, 0) * 0.3
        combined_scores[point_id] = vector_component + text_component
    
    # Sort by combined score (descending)
    ranked_ids = sorted(combined_scores.keys(), 
                       key=lambda x: combined_scores[x], 
                       reverse=True)[:TOP_K]
    
    # Create result objects with combined scores
    class HybridResult:
        def __init__(self, original_point, rrf_score):
            self.id = original_point.id
            self.payload = original_point.payload
            self.score = rrf_score
            # Copy other attributes if they exist
            if hasattr(original_point, 'vector'):
                self.vector = original_point.vector
    
    hybrid_results = []
    point_id_to_vector_result = {
        point.payload.get("source_id", str(point.id)): point 
        for point in vector_results
    }
    
    for point_id in ranked_ids:
        # Prefer vector result structure if available, otherwise use text result
        if point_id in point_id_to_vector_result:
            result = point_id_to_vector_result[point_id]
            hybrid_results.append(HybridResult(result, combined_scores[point_id]))
        elif point_id in text_results_dict:
            result = text_results_dict[point_id]['point']
            hybrid_results.append(HybridResult(result, combined_scores[point_id]))
    
    return hybrid_results

# ==============================
# RUN EVALUATION
# ==============================
vector_results = []
hybrid_results = []
ground_truth_ids = []

print("\n🚀 Running retrieval evaluation...")

for item in tqdm(eval_data):
    query = item["query"]
    answer_id = item["answer_id"]
    ground_truth_ids.append(answer_id)

    # --- Vector Search ---
    vector_res = run_vector_search(query)
    vector_results.append(vector_res)

    # --- Hybrid Search ---
    hybrid_res = run_hybrid_search(query)
    hybrid_results.append(hybrid_res)

# ==============================
# COMPUTE METRICS
# ==============================
hit_rate_vector = compute_hit_rate(vector_results, ground_truth_ids, k=TOP_K)
hit_rate_hybrid = compute_hit_rate(hybrid_results, ground_truth_ids, k=TOP_K)

print("\n===============================")
print(f"🔍 Hit Rate@{TOP_K} (Vector Search): {hit_rate_vector:.3f}")
print(f"⚡ Hit Rate@{TOP_K} (Hybrid Search): {hit_rate_hybrid:.3f}")
print("===============================")
print("✅ Retrieval evaluation completed!")