import json
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:latest"
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "json", "ftf_knowledge.json")


# ── RAG: load knowledge base once at startup ─────────────────────────────────
def load_knowledge(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load knowledge base at {path}: {e}")
        return []


KNOWLEDGE = load_knowledge(KNOWLEDGE_PATH)


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def search_knowledge(query, top_k=2):
    """Return up to top_k knowledge chunks whose words overlap with the query."""
    query_words = tokenize(query)
    scored = []
    for chunk in KNOWLEDGE:
        chunk_words = tokenize(chunk["text"])
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((overlap, chunk["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


# ── Proxy route: frontend POSTs here, we forward to Ollama ──────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json()
    if not payload or "prompt" not in payload:
        return jsonify({"error": "Request body must include a 'prompt' field."}), 400

    user_prompt = payload["prompt"]
    context_chunks = search_knowledge(user_prompt)
    if context_chunks:
        context_text = "\n".join(f"- {c}" for c in context_chunks)
        payload["prompt"] = (
            f"Use the following facts if relevant:\n{context_text}\n\n"
            f"Question: {user_prompt}"
        )

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama is not running. Start it with: ollama serve"}), 503


# ── Serve the chat UI ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── CLI mode (optional) ──────────────────────────────────────────────────────
def cli_chat():
    print("🧠 Mistral CLI Chatbot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Bye 👋")
            break
        if not user_input:
            continue
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": user_input, "stream": False},
                timeout=120,
            )
            print("Bot:", resp.json().get("response", ""))
        except requests.exceptions.ConnectionError:
            print("Bot: ❌ Cannot reach Ollama. Is it running?")


if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        cli_chat()
        print("hello guys")
    else:
        print("🚀 Server running at http://localhost:5000")
        print("   Open that URL in your browser to use the chat UI.")
        print("   Run with --cli flag for terminal mode.\n")
        app.run(port=5000, debug=False)