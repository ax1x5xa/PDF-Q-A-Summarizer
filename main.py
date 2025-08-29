# --- Imports ---
import os, tempfile, traceback
from typing import List, Tuple
from transformers import pipeline

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    try:
        from langchain_community.document_loaders.pdf import PyPDFLoader
    except Exception:
        from langchain_community.document_loaders import PyPDF2Loader as PyPDFLoader


# --- FastAPI App + Config ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Global Variables ---
qa_chain = None 
qa_history: List[Tuple[str, str]] = [] 
document_chunks = None 


# --- Build QA from PDF ---
def build_qa_from_pdf(path: str):
    global qa_chain, qa_history, document_chunks
    qa_history = []
    document_chunks = None

    loader = PyPDFLoader(path)
    documents = loader.load()
    if not documents:
        raise RuntimeError("No text extracted from PDF")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    document_chunks = docs

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.5},
    )

    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.0,
        repetition_penalty=1.05,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a careful assistant. Answer ONLY using the context below. "
            "If the answer is not in the context, reply exactly: \"I don't know\".\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

    return True


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse("<h3>Create a static/index.html file</h3>", status_code=200)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Failed to save upload: {e}"}, status_code=500)

    try:
        build_qa_from_pdf(tmp_path)
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Failed to process PDF: {e}"}, status_code=500)

    return {"success": True, "message": "PDF processed successfully"}

@app.post("/summarize")
async def summarize_document():
    global document_chunks
    if not document_chunks:
        return JSONResponse({"success": False, "error": "No PDF processed yet"}, status_code=400)

    try:
        raw_text = " ".join([chunk.page_content for chunk in document_chunks[:5]]).strip()
        if not raw_text:
            return JSONResponse({"success": False, "error": "No extractable text found in PDF"}, status_code=400)

        text_to_summarize = raw_text[:3000]   # keep it safe under model input limit
        print(f"📝 Summarizing {len(text_to_summarize)} chars")

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
        )
        summary_out = summarizer(text_to_summarize, max_length=200, min_length=60, do_sample=False)
        summary = summary_out[0]["summary_text"]
    except Exception as e:
        print("❌ Full summarization traceback:")
        traceback.print_exc()
        return JSONResponse({"success": False, "error": f"Summarization error: {str(e)}"}, status_code=500)

    return {"success": True, "summary": summary}

@app.post("/ask")
async def ask_question(q: str = Form(...)):
    global qa_chain, qa_history
    if qa_chain is None:
        return JSONResponse({"success": False, "error": "No PDF processed yet"}, status_code=400)

    try:
        result = qa_chain.invoke({"query": q})
        answer = result["result"]
        sources = result.get("source_documents", [])
    except Exception as e:
        print("❌ QA chain error:")
        traceback.print_exc()
        return JSONResponse({"success": False, "error": f"QA chain error: {e}"}, status_code=500)

    if not answer or answer.strip() in ["", "No answer found."]:
        answer = "I don't know"

    qa_history.append((q, answer))

    def _short(doc: Document, n=220):
        txt = (doc.page_content or "").strip().replace("\n", " ")
        return txt[:n] + ("…" if len(txt) > n else "")
    src_payload = [
        {"page": (doc.metadata.get("page") if isinstance(doc.metadata, dict) else None),
         "snippet": _short(doc)}
        for doc in (sources or [])
    ]

    return {"success": True, "answer": answer, "history": qa_history, "sources": src_payload}

@app.get("/history")
async def get_history():
    return {"history": qa_history}
