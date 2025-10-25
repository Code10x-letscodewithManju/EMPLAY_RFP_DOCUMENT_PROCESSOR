import os
import re
import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from bs4 import BeautifulSoup

# Embeddings & FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Google Gemini via LangChain
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# -------------------------
# Load environment
# -------------------------
load_dotenv()
BASE_DIR = Path.cwd()
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(exist_ok=True)
META_PATH = VECTORSTORE_DIR / "meta.json"
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# -------------------------
# Config
# -------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))  # increased
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 4))

STRUCTURED_FIELDS = [
    "Bid Number", "Title", "Due Date", "Bid Submission Type", "Term of Bid",
    "Pre Bid Meeting", "Installation", "Bid Bond Requirement", "Delivery Date",
    "Payment Terms", "Any Additional Documentation Required", "MFG for Registration",
    "Contract or Cooperative to use", "Model_no", "Part_no", "Product",
    "contact_info", "company_name", "Bid Summary", "Product Specification"
]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")

# -------------------------
# Text extraction
# -------------------------
def extract_text_from_pdf(filepath: str) -> str:
    parts = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                parts.append(txt)
    return "\n".join(parts).strip()

def extract_text_from_html(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    text = "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())
    return text

def load_text(filepath: str) -> str:
    fp = str(filepath)
    if fp.lower().endswith(".pdf"):
        return extract_text_from_pdf(fp)
    elif fp.lower().endswith((".html", ".htm")):
        return extract_text_from_html(fp)
    else:
        raise ValueError("Unsupported file type. Only .pdf, .html, .htm allowed.")

# -------------------------
# Chunking
# -------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = end - overlap
    return chunks

# -------------------------
# Local FAISS Vector Store
# -------------------------
class LocalVectorStore:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []
        if INDEX_PATH.exists() and META_PATH.exists():
            self._load()
        else:
            self.index = None
            self.meta = []

    def _save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _load(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def add_documents(self, chunks: List[str], source: str):
        if not chunks:
            return
        embs = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        if self.index is None:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        start_id = len(self.meta)
        for i, chunk in enumerate(chunks):
            self.meta.append({"source": source, "chunk_id": start_id + i, "text": chunk})
        self._save()

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append((float(score), self.meta[idx]))
        return results

# -------------------------
# Google Gemini LLM wrapper
# -------------------------
def get_gemini_llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain_google_genai not installed.")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    class LLMWrapper:
        def __init__(self, llm_obj):
            self.llm_obj = llm_obj
        def invoke(self, prompt: str):
            resp = self.llm_obj.invoke(prompt)
            return getattr(resp, "content", str(resp))
    return LLMWrapper(llm)

# -------------------------
# Prompt builders
# -------------------------
def build_extraction_prompt(text: str) -> str:
    header = (
        "You are an expert RFP document analyst and structured data extractor. "
       "Analyze the document carefully and extract all required fields into STRICTLY valid JSON. "
       "If a field is missing, infer logically or use 'N/A'.\n\n"
       "Ensure correctness, conciseness, and include all fields.\n\n"
    )
    fields_list = "\n".join(f"- {f}" for f in STRUCTURED_FIELDS)
    instruction = f"""
Document text (begin):
{text[:15000]}
Document text (end)

Return ONLY valid JSON (no commentary, no markdown fences).
"""
    example_dict = {f: "N/A" for f in STRUCTURED_FIELDS}
    example_json = json.dumps(example_dict, indent=2, ensure_ascii=False)
    return header + fields_list + instruction + example_json

def build_rag_answer_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n---\n".join(contexts)
    prompt = (
        "You are an assistant answering a question using ONLY the provided context. "
        "If answer not present, say 'Answer not found in context.' "
        "Provide a concise answer and mention filename(s) supporting the facts.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

# -------------------------
# Core processing functions
# -------------------------
def ingest_and_index(filepath: str, vs: LocalVectorStore) -> Tuple[str, int]:
    text = load_text(filepath)
    chunks = chunk_text(text)
    vs.add_documents(chunks, source=Path(filepath).name)
    return text, len(chunks)

def extract_structured_json(filepath: str, llm_wrapper) -> dict:
    text = load_text(filepath)
    prompt = build_extraction_prompt(text)
    resp = llm_wrapper.invoke(prompt)
    if not isinstance(resp, str):
        resp = str(resp)
    try:
        json_match = re.search(r'\{.*\}', resp, re.DOTALL)
        if not json_match:
            cleaned = re.sub(r'```.*?```', '', resp, flags=re.DOTALL)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        resp_text = json_match.group(0)
        data = json.loads(resp_text)
    except Exception:
        raise RuntimeError(f"LLM did not return valid JSON. Response: {resp[:2000]}")
    for k in STRUCTURED_FIELDS:
        if k not in data or not data[k]:
            data[k] = "N/A"
    out_file = OUTPUT_DIR / f"{Path(filepath).stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def rag_answer(question: str, vs: LocalVectorStore, llm_wrapper, top_k: int = TOP_K) -> dict:
    results = vs.search(question, top_k)
    if not results:
        return {"answer": "No documents indexed yet. Please ingest documents first.", "sources": []}
    contexts = [f"Source {r[1]['source']}\n{r[1]['text']}" for r in results]
    prompt = build_rag_answer_prompt(question, contexts)
    resp = llm_wrapper.invoke(prompt)
    return {"answer": resp, "sources": [r[1] for r in results]}

# -------------------------
# Streamlit UI
# -------------------------
# --------------------------------------------
# 🌟 APP HEADER & PAGE CONFIGURATION
# --------------------------------------------
st.set_page_config(
    page_title="RFP DOCUMENT PROCESSOR - LLM + RAG",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.markdown("""
<div style="text-align:center; padding: 20px 0;">
    <h1 style="color:#1E88E5;">🕵️‍♂️RFP DOCUMENT PROCESSOR - LLM + RAG</h1>
   <h3 style="color:#009688;">LLM & RAG-Powered Extraction & Structuring of Procurement Documents</h3>
    <p style="color:#FFFFFF; font-size:16px; margin-top:10px;">
        Upload your RFP (PDF or HTML), and let our Intelligent Engine powered by 
        <b>LLM</b> + <b>FAISS RAG</b> transform it into a clean, structured JSON format — 
        ready for analysis, reporting, or integration.
    </p>
</div>
""", unsafe_allow_html=True)


if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Put it in a `.env` file or your environment.")
    st.stop()

vs = LocalVectorStore()
try:
    llm = get_gemini_llm()
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {e}")
    st.stop()

# -------------------------
# File upload & ingestion
# -------------------------
st.sidebar.header("Upload & Ingest")
uploaded = st.sidebar.file_uploader("Upload PDF/HTML", type=["pdf", "html", "htm"])
if uploaded:
    temp_path = TEMP_DIR / uploaded.name
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.sidebar.success(f"Saved to {temp_path}")
    if st.sidebar.button("Ingest (add to RAG index)"):
        with st.spinner("Extracting text and indexing..."):
            try:
                text, nchunks = ingest_and_index(str(temp_path), vs)
                st.sidebar.success(f"Ingested {nchunks} chunks from {uploaded.name}")
                st.sidebar.markdown("Preview (first 500 chars):")
                st.sidebar.code(text[:500])
            except Exception as e:
                st.sidebar.error(f"Ingest error: {e}")

# -------------------------
# Structured Extraction
# -------------------------
st.header("Structured Extraction (LLM + RAG)")
sources = sorted({m["source"] for m in vs.meta}) if vs.meta else []
selected_source = st.selectbox("Choose source to extract from", options=["-- select --"] + sources)
if st.button("Extract structured JSON from selected source"):
    if selected_source == "-- select --":
        st.warning("Select an ingested source first.")
    else:
        matched_paths = list(TEMP_DIR.glob(selected_source))
        if matched_paths:
            file_path = matched_paths[0]
            with st.spinner("Calling Gemini to extract structured JSON..."):
                try:
                    data = extract_structured_json(str(file_path), llm)
                    st.success("Extraction completed. JSON saved to outputs")
                    st.json(data)
                    st.download_button("Download JSON", json.dumps(data, indent=2), file_name=f"{Path(file_path).stem}.json", mime="application/json")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
        else:
            st.error("Original file not found in temp; re-upload and ingest.")

# -------------------------
# Quick Extract (no indexing)
# -------------------------
st.markdown("---")
st.subheader("Quick Extraction (LLM only, no indexing)")
quick_file = st.file_uploader("Upload a single file for quick extract", type=["pdf", "html", "htm"], key="quick")
if quick_file:
    qpath = TEMP_DIR / quick_file.name
    with open(qpath, "wb") as f:
        f.write(quick_file.getbuffer())
    st.success(f"Saved {qpath}")
    if st.button("Quick Extract now"):
        with st.spinner("Calling Gemini to extract JSON..."):
            try:
                data = extract_structured_json(str(qpath), llm)
                st.success("Quick extraction done.")
                st.json(data)
                st.download_button("Download JSON", json.dumps(data, indent=2), file_name=f"{Path(qpath).stem}.json", mime="application/json")
            except Exception as e:
                st.error(f"Quick extract failed: {e}")

# --------------------------------------------------
# 2. GLOBAL STYLES — APP THEME
# --------------------------------------------------
st.markdown("""
<style>
/* Global Page */
body {
    background-color: #F5F9FF;
    font-family: 'Poppins', sans-serif;
}


/* Upload Card */
.upload-card {
    background: linear-gradient(135deg, #E3F2FD, #E1F5FE);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1976D2, #2196F3);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 15px;
    transition: 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2196F3, #42A5F5);
    transform: scale(1.02);
}

/* JSON Output Box */
.output-box {
    background: #FFFFFF;
    border: 1px solid #E3F2FD;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    font-family: monospace;
    color: #263238;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #607D8B;
    font-size: 14px;
}

/* Gradient Divider */
hr {
    border: 0;
    height: 2px;
    background: linear-gradient(to right, #90CAF9, #4DD0E1);
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)




# -------------------------------
# 📄 Project Information Footer (Clean Version)
# -------------------------------
import streamlit as st

# --- Simple, professional style ---
st.markdown("""
<style>
.project-summary {
    background-color: #F8FBFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 20px 25px;
    margin-top: 25px;
    font-family: 'Poppins', sans-serif;
}
.project-summary h3 {
    color: #1565C0;
    margin-bottom: 10px;
}
.project-summary h4 {
    color: #0D47A1;
    margin-top: 15px;
    margin-bottom: 8px;
}
.project-summary p, .project-summary li {
    color: #1A1A1A;
    font-size: 15px;
    line-height: 1.6;
}
.project-summary ul {
    margin-top: 5px;
    padding-left: 20px;
}
.note {
    color: #424242;
    font-style: italic;
    font-size: 14px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# 📄 Project Information Footer (Plain Version)
# -------------------------------

st.markdown("---")
st.markdown("### 🧾 Project Summary")
st.write("""
RFP Processor — LLM + RAG

This Streamlit app extracts and structures key information from RFP documents (PDF/HTML)
into a clean JSON format using Google Gemini LLM and a FAISS-powered vector store.

Key Features:
• Automatic field extraction from uploaded RFPs  
• Document chunking and embedding using SentenceTransformers  
• Local FAISS vector store for efficient retrieval (RAG-ready)  
• Structured JSON export for evaluation  

Tech Stack:
• Streamlit (Frontend & UI)  
• LangChain + Google Gemini (LLM Integration)  
• FAISS (Vector Storage)  
• PyPDF2 / BeautifulSoup4 (Parsing)  
• SentenceTransformers (Embeddings)   

Outputs are saved in the 'outputs' folder.  
FAISS index and metadata are stored in 'vectorstore'.
""")
st.markdown("---")
