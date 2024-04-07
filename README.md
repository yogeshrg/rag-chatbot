# RAG Chatbot

A local, privacy-first RAG (Retrieval-Augmented Generation) chatbot built with **LlamaIndex**, **Ollama**, and **FastAPI**. Runs entirely on your machine — no API keys, no cloud dependencies.

## Features

- 🔒 **100% Local** — Uses Ollama for LLM and embeddings, no external APIs
- 📚 **Document Q&A** — Upload PDFs, Markdown, code files, and chat with them
- 🎯 **Hybrid Retrieval** — Vector search with similarity threshold filtering
- ⚡ **Streaming Responses** — Real-time token streaming via SSE
- 🐳 **Docker Ready** — One-command deployment with docker-compose
- 🎨 **Streamlit UI** — Beautiful chat interface included
- 🧪 **Tested** — Pytest suite with unit and integration tests

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone and navigate
cd rag-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,ui]"

# Copy environment config
cp .env.example .env

# Start Ollama and pull models
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Run API server
python -m app.main

# In another terminal, run Streamlit UI
streamlit run app/ui/streamlit_app.py
```

### Docker Deployment (Recommended)

```bash
cd docker
docker-compose up -d

# Wait for models to download (check logs)
docker-compose logs -f model-puller

# Access:
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## Usage

### CLI

```bash
# Interactive chat
python -m app.cli chat

# Single question
python -m app.cli chat "What is RAG?"

# Ingest documents
python -m app.cli ingest ./data/documents -r

# List indexed documents
python -m app.cli list-docs

# Health check
python -m app.cli health
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/chat` | Non-streaming chat |
| POST | `/api/v1/chat/stream` | Streaming chat (SSE) |
| POST | `/api/v1/documents/upload` | Upload single document |
| POST | `/api/v1/documents/upload/batch` | Upload multiple documents |
| GET | `/api/v1/documents` | List indexed documents |
| DELETE | `/api/v1/documents/{id}` | Delete document |
| POST | `/api/v1/documents/reindex` | Rebuild entire index |

### Example API Usage

```bash
# Chat (streaming)
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG", "stream": true}'

# Upload document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│   LlamaIndex │
│     UI      │     │    API      │     │   Pipeline   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐            │
                    │   Ollama    │◀───────────┘
                    │  (LLM +     │
                    │ Embeddings) │
                    └─────────────┘
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | LLM model for generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `TOP_K` | `5` | Documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score |

## Project Structure

```
rag-chatbot/
├── app/
│   ├── api/routes.py          # FastAPI routes
│   ├── cli.py                 # Typer CLI
│   ├── core/config.py         # Pydantic settings
│   ├── main.py                # FastAPI app entry
│   ├── models/schemas.py      # Pydantic models
│   ├── services/rag_service.py # RAG pipeline
│   └── ui/streamlit_app.py    # Streamlit UI
├── docker/
│   ├── Dockerfile             # Multi-stage build
│   └── docker-compose.yml     # Full stack
├── tests/                     # Pytest suite
├── pyproject.toml             # Project config
└── .env.example               # Environment template
```

## Supported File Types

- PDF (`.pdf`)
- Text/Markdown (`.txt`, `.md`)
- Code (`.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`)

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_rag_chatbot.py::TestRAGService -v
```

## Customization

### Change Models

```bash
# Edit .env
OLLAMA_LLM_MODEL=mistral:7b
OLLAMA_EMBED_MODEL=mxbai-embed-large

# Pull new models
ollama pull mistral:7b
ollama pull mxbai-embed-large
```

### Add Reranking

Install cross-encoder and modify `rag_service.py`:

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank
# or local: from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
```

## Troubleshooting

**Ollama connection refused**
```bash
# Ensure Ollama is running
ollama serve
```

**Out of memory**
```bash
# Use smaller models
OLLAMA_LLM_MODEL=llama3.1:8b-instruct-q4_0
```

**Slow responses**
- Reduce `CHUNK_SIZE` and `TOP_K`
- Use quantized models (q4_0, q4_K_M)
- Enable GPU acceleration for Ollama

## License

MIT License — see LICENSE file for details.

## Author

Yogesh Gaikwad — Yogesh.r.gaikwad@gmail.com