#!/usr/bin/env python3
"""
Aura Gemini Bridge — OpenAI-to-Gemini Proxy
Accepts OpenAI-format requests and forwards them to Google's OpenAI-compatible
bridge endpoint with the Gemini API key injected automatically.

This allows Letta (and any OpenAI-compatible client) to use Gemini 2.5 Flash
without needing to configure API keys in the client.
"""
import os
import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_BRIDGE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
PORT = int(os.environ.get("PORT", 8080))

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "aura-gemini-bridge",
        "key_configured": bool(GEMINI_API_KEY),
        "bridge": GOOGLE_BRIDGE_URL
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available Gemini models in OpenAI format."""
    try:
        resp = requests.get(
            f"{GOOGLE_BRIDGE_URL}/models",
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
            timeout=15
        )
        return Response(resp.content, status=resp.status_code,
                        content_type="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Proxy chat completions to Gemini with injected API key."""
    try:
        body = request.get_json(force=True)
        model = body.get("model", "gemini-2.5-flash")
        is_stream = body.get("stream", False)

        print(f"[BRIDGE] {model} | messages={len(body.get('messages', []))} | stream={is_stream}")

        resp = requests.post(
            f"{GOOGLE_BRIDGE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=180,
            stream=is_stream
        )

        print(f"[BRIDGE] Response: {resp.status_code}")

        if is_stream:
            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    yield chunk
            return Response(
                generate(),
                status=resp.status_code,
                content_type=resp.headers.get("content-type", "text/event-stream")
            )
        else:
            return Response(
                resp.content,
                status=resp.status_code,
                content_type="application/json"
            )
    except Exception as e:
        print(f"[BRIDGE] Error: {e}")
        return jsonify({"error": {"message": str(e), "type": "proxy_error"}}), 500


@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy_all(path):
    """Catch-all proxy for any other OpenAI API endpoints."""
    try:
        url = f"{GOOGLE_BRIDGE_URL}/{path}"
        resp = requests.request(
            method=request.method,
            url=url,
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request.get_json(silent=True),
            timeout=60
        )
        return Response(resp.content, status=resp.status_code,
                        content_type="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Aura Gemini Bridge starting on port {PORT}")
    print(f"Key configured: {bool(GEMINI_API_KEY)}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
