# rag-api-service

Setup:

```powershell
uv sync
```

Create a `.env` file from `.env.example` and set `OPENAI_API_KEY`.

Create a `/documents` folder in the project root. This folder can contain PDF files and is intended for local testing only.

Qdrant modes:

- `QDRANT_MODE=local` stores data in `./qdrant_data` and does not need Docker.
- `QDRANT_MODE=server` connects to `QDRANT_HOST` and `QDRANT_PORT`.
- `QDRANT_MODE=url` connects to `QDRANT_URL` and optionally `QDRANT_API_KEY` for Qdrant Cloud.

Recommended local setup without Docker:

```powershell
$env:QDRANT_MODE="local"
uv run --env-file .env python .\main\main.py
```

If you prefer a Docker Qdrant server:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
$env:QDRANT_MODE="server"
uv run --env-file .env python .\main\main.py
```
