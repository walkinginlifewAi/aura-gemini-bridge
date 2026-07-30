# Aura Gemini Bridge

OpenAI-to-Gemini proxy for the Aura Nexus system.

Accepts OpenAI-format requests and forwards them to Google's OpenAI-compatible bridge with the Gemini API key injected automatically. This allows Letta to use Gemini 2.5 Flash without needing to configure API keys inside the Letta container.

## Environment Variables

- `GEMINI_API_KEY` — Your Google Gemini API key (required)
- `PORT` — Port to listen on (default: 8080)

## Endpoints

- `GET /health` — Health check
- `GET /v1/models` — List available Gemini models
- `POST /v1/chat/completions` — Chat completions (proxied to Gemini)
- `* /v1/*` — All other OpenAI API endpoints proxied

## Letta Configuration

Set Letta's openai provider endpoint to `http://<this-service-host>:8080/v1/` with any placeholder API key.
