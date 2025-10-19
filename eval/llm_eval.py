import sys
from pathlib import Path
import json
from time import sleep
import json
from qdrant_client import QdrantClient
from tqdm import tqdm
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# ------------------------
# Paths and Imports
# ------------------------
# Add src folder to sys.path to import rag_pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Import your RAG answer generation function
from rag_pipeline import get_answer  # <-- make sure this matches your function

# ------------------------
# Azure OpenAI Config
# ------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = "text-embedding-3-small"

MODELS = ["gpt-4o", "gpt-4o-mini"]  # Azure deployment names

client_openai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# ------------------------
# Load queries
# ------------------------
with open("../retrieval_eval_ground_truth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# ------------------------
# LLM Judge Function
# ------------------------
def judge_answer(model_name, query, answer):
    """
    Ask the LLM to judge whether the answer is relevant to the query.
    Returns "Relevant" or "Not Relevant".
    """
    prompt = f"""
You are an evaluation assistant. Your task is to compare the following question and answer.

Question: {query}
Answer: {answer}

Determine if the answer is relevant to the question. Only reply with one of these words: "Relevant" or "Not Relevant".
"""
    try:
        response = client_openai.chat.completions.create(
            deployment_name=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        judgement = response.choices[0].message.content.strip()
        return judgement
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        return "Error"

# ------------------------
# Run evaluation
# ------------------------
results = {model: [] for model in MODELS}

for item in ground_truth:
    query = item["query"]
    
    # 1️⃣ Generate answer using RAG pipeline
    answer = get_answer(query)
    if not answer:
        print(f"No answer generated for query: {query}")
        continue

    # 2️⃣ Ask LLM judge for each model
    for model in MODELS:
        judgement = judge_answer(model, query, answer)
        results[model].append({
            "query": query,
            "answer": answer,
            "judgement": judgement
        })
        # avoid throttling
        sleep(0.5)

# ------------------------
# Save results
# ------------------------
for model in MODELS:
    output_file = f"llm_judge_results_{model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results[model], f, ensure_ascii=False, indent=2)
    print(f"Saved results for {model} to {output_file}")
