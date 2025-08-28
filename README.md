# PDF Q&A + Summarizer (FastAPI)

Simple FastAPI app that lets you upload a PDF, ask questions using a vector retriever + FLAN-T5, and generate a summary with BART.

## Local Dev

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --reload
```

Open: <http://localhost:8000>

## Notes

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via `HuggingFaceEmbeddings`).
- Retriever: FAISS with MMR (k=6, fetch_k=30).
- LLM for Q&A: `google/flan-t5-base` (transformers pipeline).
- Summarization: `facebook/bart-large-cnn`.

**Compatibility tip:** Depending on your `langchain` version, the `HuggingFaceEmbeddings` import may live in `langchain_huggingface` (newer) or `langchain.embeddings` (older). The code handles both.

## Deploy (optional)

### Docker
```Dockerfile
# See Dockerfile in this repo
```
Build & run:
```bash
docker build -t pdf-qa-app .
docker run -p 8000:8000 pdf-qa-app
```

### Run without Docker
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

- `POST /upload`  — form-data file upload under key `file` (PDF).
- `POST /summarize` — summarizes the first few chunks.
- `POST /ask` — form-encoded field `q=your question`.
- `GET /history` — returns Q&A pairs.

## License
MIT © 2025
