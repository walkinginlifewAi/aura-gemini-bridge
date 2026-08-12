#!/usr/bin/env python3
"""
Aura Gemini Bridge — OpenAI-to-Gemini Proxy
Accepts OpenAI-format requests and forwards them to Google's OpenAI-compatible
bridge endpoint with the Gemini API key injected automatically.

This allows Letta (and any OpenAI-compatible client) to use Gemini 2.5 Flash
without needing to configure API keys in the client.
"""
import os
import threading
import time
import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_BRIDGE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
PORT = int(os.environ.get("PORT", 8080))
MAX_CONCURRENT_REQUESTS = max(1, int(os.environ.get("MAX_CONCURRENT_REQUESTS", "1")))
DEFAULT_RATE_LIMIT_SECONDS = max(5, int(os.environ.get("DEFAULT_RATE_LIMIT_SECONDS", "60")))
QUEUE_WAIT_SECONDS = max(15, int(os.environ.get("QUEUE_WAIT_SECONDS", "210")))
REQUEST_GATE = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
COOLDOWN_LOCK = threading.Lock()
COOLDOWN_UNTIL = 0.0

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")


@app.route("/health", methods=["GET"])
def health():
    with COOLDOWN_LOCK:
        cooldown_seconds = max(0, round(COOLDOWN_UNTIL - time.monotonic()))
    return jsonify({
        "status": "ok",
        "service": "aura-gemini-bridge",
        "key_configured": bool(GEMINI_API_KEY),
        "bridge": GOOGLE_BRIDGE_URL,
        "cooldown_seconds": cooldown_seconds,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
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
    global COOLDOWN_UNTIL
    with COOLDOWN_LOCK:
        remaining = COOLDOWN_UNTIL - time.monotonic()
    if remaining > 0:
        return jsonify({
            "error": {
                "message": "Gemini bridge is cooling down after an upstream rate limit. Retry after the stated delay.",
                "type": "rate_limit_cooldown",
                "retry_after_seconds": round(remaining),
            }
        }), 429
    if not REQUEST_GATE.acquire(blocking=True, timeout=QUEUE_WAIT_SECONDS):
        return jsonify({
            "error": {
                "message": "Aura is still processing a queued request. Retry after the stated delay.",
                "type": "bridge_queue_timeout",
                "retry_after_seconds": QUEUE_WAIT_SECONDS,
            }
        }), 429
    try:
        body = request.get_json(force=True)
        model = body.get("model", "gemini-2.5-flash")
        is_stream = body.get("stream", False)

        # Gemini's OpenAI-compatible endpoint can return a malformed empty
        # function-call completion when explicit "tool_choice": "auto" is
        # supplied. Letta's default is automatic selection, which Gemini
        # already performs when the field is omitted.
        proxy_body = dict(body)
        if proxy_body.get("tools") and proxy_body.get("tool_choice") == "auto":
            proxy_body.pop("tool_choice", None)
            print("[BRIDGE] Removed explicit auto tool_choice for Gemini compatibility")

        print(f"[BRIDGE] {model} | messages={len(body.get('messages', []))} | stream={is_stream}")

        resp = requests.post(
            f"{GOOGLE_BRIDGE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=proxy_body,
            timeout=180,
            stream=is_stream
        )

        print(f"[BRIDGE] Response: {resp.status_code}")
        if resp.status_code == 200 and not is_stream:
            try:
                completion = resp.json()
                choice = (completion.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                print(
                    "[BRIDGE] Completion shape: "
                    f"finish_reason={choice.get('finish_reason')} "
                    f"content_length={len(message.get('content') or '')} "
                    f"tool_calls={len(message.get('tool_calls') or [])}"
                )
            except (ValueError, TypeError, AttributeError):
                print("[BRIDGE] Completion shape unavailable")
        if resp.status_code == 429:
            try:
                retry_after = max(DEFAULT_RATE_LIMIT_SECONDS, int(resp.headers.get("Retry-After", "0")))
            except ValueError:
                retry_after = DEFAULT_RATE_LIMIT_SECONDS
            with COOLDOWN_LOCK:
                COOLDOWN_UNTIL = time.monotonic() + retry_after
            print(f"[BRIDGE] Upstream rate limited; cooldown={retry_after}s; detail={resp.text[:300]}")

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
    finally:
        REQUEST_GATE.release()


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
