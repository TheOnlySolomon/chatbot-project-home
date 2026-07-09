import requests
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder=".")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:latest"


# ── Proxy route: frontend POSTs here, we forward to Ollama ──────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json()
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
    else:
        print("🚀 Server running at http://localhost:5000")
        print("   Open that URL in your browser to use the chat UI.")
        print("   Run with --cli flag for terminal mode.\n")
        app.run(port=5000, debug=False)