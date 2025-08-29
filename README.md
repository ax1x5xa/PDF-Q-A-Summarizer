# PDF Q&A + Summarizer (FastAPI)

Simple FastAPI app that lets you upload a PDF, ask questions using a vector retriever + FLAN-T5, and generate a summary with BART.

## Features

- Upload PDF files
- Extract & chunk text from PDF
- Store embeddings with FAISS
- Summarize document
- Ask questions about the content
- Uses Hugging Face models for embeddings, responses and summarization

## Notes

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (via `HuggingFaceEmbeddings`).
- Retriever: FAISS with MMR (k=6, fetch_k=30).
- LLM for Q&A: `google/flan-t5-base` (transformers pipeline).
- Summarization: `facebook/bart-large-cnn`.

**Compatibility tip:** Depending on your `langchain` version, the `HuggingFaceEmbeddings` import may live in `langchain_huggingface` (newer) or `langchain.embeddings` (older). The code handles both.

## Requirements
- Python 3.9–3.11
- (Optional) A [Hugging Face access token](https://huggingface.co/settings/tokens)  
  > You’ll need this if you want to use gated/private models or swap in other Hugging Face models that require authentication.

---

## Installation
Clone the repo:
```bash
git clone https://github.com/ax1x5xa/PDF-Q-A-Summarizer.git
cd pdf-qa-summarizer
```

Create virtual environment (optional but recommended):
```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app (local dev / auto-reload):
```bash
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## Deploy (optional)

### Docker
```Dockerfile
# See Dockerfile in this repo
```
Build & run:
```bash
docker build -t pdf-qa-summarizer .
docker run -p 8000:8000 pdf-qa-summarizer
```

### Run without Docker
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API
- `POST /upload`  — form-data file upload under key `file` (PDF).
- `POST /summarize` — summarizes the first few chunks.
- `POST /ask` — form-encoded field `q=your question`.
- `GET /history` — returns Q&A pairs.

## Project Structure
```
pdf-qa-summarizer/
├─ main.py
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
├─ README.md
├─ LICENSE
└─ static/
   └─ index.html
```

---

## Acknowledgments
- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [FastAPI](https://fastapi.tiangolo.com/)

## Connect with me
- [LinkedIn](https://www.linkedin.com/in/x1x5x/)
- [Kaggle](https://www.kaggle.com/abdullahhussein1504)

## License
MIT © 2025
