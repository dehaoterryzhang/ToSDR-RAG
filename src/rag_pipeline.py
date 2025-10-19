from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

TOP_K = 3  # number of top relevant documents to retrieve
embed_model = "text-embedding-3-small"
gpt_model = "gpt-4o"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QDRANT_PATH = os.path.join(PROJECT_ROOT, "data/qdrant_data")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

client_openai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# -----------------------------
# Function: Search Qdrant
# -----------------------------
def search_documents(query_embedding: List[float], top_k: int = TOP_K):
    """
    Search Qdrant collection for top_k most similar documents.
    Returns a list of hits with payload.
    """

    client_qdrant = QdrantClient(path=QDRANT_PATH)

    hits = client_qdrant.search(
        collection_name="tosdr_docs",
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return hits

# -----------------------------
# Function: Build Prompt
# -----------------------------
def build_prompt(hits: List[PointStruct], query: str) -> str:
    """
    Build a RAG prompt template using top hits.
    """
    context_sections = []
    for hit in hits:
        source = hit.payload.get("source", "unknown")
        content = hit.payload.get("content", "")
        section = f"SOURCE: {source}\nCONTENT: {content}"
        context_sections.append(section)
    
    context = "\n\n".join(context_sections)
    
    prompt = f"""
You are an intelligent assistant that helps users answer questions about Terms of Service and Privacy Policies.

Answer the QUESTION based on the relevant documents retrieved from the TOSDR database.
Use only the facts from the content when answering the QUESTION.

{context}

QUESTION: {query}
"""
    return prompt

# -----------------------------
# Function: Call LLM
# -----------------------------
def call_llm(client_openai: AzureOpenAI, prompt: str) -> str:
    """
    Call Azure OpenAI to generate an answer from the prompt.
    """
    response = client_openai.chat.completions.create(
        model=gpt_model, 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -----------------------------
# Function: Orchestrator
# -----------------------------
def rag(query: str) -> str:
    """
    Full RAG pipeline: embed query, search, build prompt, call LLM.
    """
    # Step 1: Embed the query
    embedding_resp = client_openai.embeddings.create(
        model=embed_model,
        input=query
    )
    query_embedding = embedding_resp.data[0].embedding

    # Step 2: Search top documents
    hits = search_documents(query_embedding, top_k=TOP_K)

    if not hits:
        return "No relevant documents found."

    # Step 3: Build prompt
    prompt = build_prompt(hits, query)

    # Step 4: Call LLM
    answer = call_llm(client_openai, prompt)
    return answer

if __name__ == "__main__":

    query = "Does apple allow tracking cookies?"
    answer = rag(query)
    print("=== Answer ===")
    print(answer)