from flask import Flask, render_template, request, jsonify
from rag_pipeline import rag  # your RAG pipeline
import os
import json
import datetime

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
app = Flask(__name__, template_folder=template_dir)

# Create feedback log directory if it doesn't exist
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(log_dir, exist_ok=True)
feedback_log_file = os.path.join(log_dir, "feedback.jsonl")

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        if query.strip():
            answer = rag(query)
    return render_template("index.html", query=query, answer=answer)

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json()
        query = data.get("query", "")
        answer = data.get("answer", "")
        rating = data.get("rating", "")  # "thumbs_up" or "thumbs_down"
        
        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "rating": rating
        }
        
        # Log feedback to file
        with open(feedback_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        return jsonify({"status": "success", "message": "Feedback recorded successfully"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)