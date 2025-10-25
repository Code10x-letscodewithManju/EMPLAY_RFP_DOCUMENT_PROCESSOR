
---

# 📄 RFP Document Processor — LLM + RAG (JSON Extraction Only)

## 🧠 Project Overview

The **RFP Processor** is an intelligent application designed to extract **structured information** from **RFP (Request for Proposal)** documents — whether in **PDF** or **HTML** format — into a **strict JSON structure**.

This system combines the power of **Google Gemini (via LangChain)** for natural language understanding with a **local RAG (Retrieval-Augmented Generation) pipeline** for scalable document storage and retrieval.

Although the **Q&A retrieval interface** is not included, all embeddings and metadata are stored locally, enabling future expansion.

---

### 🔍 Core Capabilities

1. Upload and process **PDF/HTML** RFP documents.
2. **Extract**, **chunk**, and **embed** text using **SentenceTransformers**.
3. Store embeddings locally in a **FAISS vectorstore** (RAG-ready).
4. Use **Google Gemini LLM** for **precise JSON extraction** of RFP fields.
5. Supports two processing modes:

   * **Structured Extraction (LLM + RAG)** → Uses FAISS index and LLM.
   * **Quick Extraction (LLM-only)** → Directly extracts JSON without indexing.
6. Save JSON results to the `outputs/` folder for analysis or evaluation.

---

## ⚙️ Dependencies

Install the required dependencies using:

```bash
pip install streamlit PyPDF2 beautifulsoup4 python-dotenv sentence-transformers faiss-cpu langchain-google-genai
```

**Package Breakdown:**

| Package                  | Purpose                                                |
| ------------------------ | ------------------------------------------------------ |
| `streamlit`              | Interactive UI for document upload and JSON extraction |
| `PyPDF2`                 | Extract text from PDF documents                        |
| `beautifulsoup4`         | Parse and clean HTML-based RFPs                        |
| `python-dotenv`          | Load environment variables (API keys, config)          |
| `sentence-transformers`  | Generate vector embeddings for text                    |
| `faiss-cpu`              | Local FAISS index for RAG vector storage               |
| `langchain-google-genai` | Integrate Google Gemini for LLM-based extraction       |

---

## 📂 Project Structure

```
RFP-Processor/
│
├─ app.py                   # Main Streamlit application
├─ .env                     # Environment variables (API keys)
├─ outputs/                 # JSON outputs of extracted RFP data
├─ temp/                    # Temporary upload storage
├─ vectorstore/             # FAISS index and metadata
│   ├─ faiss.index          # FAISS embedding index
│   └─ meta.json            # Chunk metadata
└─ README.md                # Project documentation
```

---

## 🔧 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Code10x-letscodewithManju/EMPLAY_RFP_DOCUMENT_PROCESSOR.git
cd EMPLAY_RFP_DOCUMENT_PROCESSOR
```

### 2️⃣ Configure environment variables

Create a `.env` file in the root folder:

```text
GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
GOOGLE_GEMINI_MODEL=gemini-2.0-flash
```

| Variable              | Description                           |
| --------------------- | ------------------------------------- |
| `GOOGLE_API_KEY`      | Your Google API key for Gemini access |
| `GOOGLE_GEMINI_MODEL` | The Gemini model used for extraction  |

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

*(or use the pip install command mentioned above)*

---

## 🚀 Running the App

Launch the Streamlit application using:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal, usually:
👉 `http://localhost:8501`

---

## 🧩 Application Flow

### **1. Upload & Process RFP Documents**

* Upload one or multiple **PDF/HTML** RFPs.
* The system extracts text, splits it into **chunks**, and generates **embeddings** using `sentence-transformers`.
* All embeddings and metadata are saved in the `vectorstore/` directory for **RAG integration**.

---

### **2. Structured Extraction (LLM + RAG)**

* Select a document source from the dropdown.
* Click **Extract Structured JSON**.
* The LLM uses document context and extracted embeddings to create a **precise JSON** containing all required RFP fields.

**Generated JSON Template:**

```json
{
  "Bid Number": "",
  "Title": "",
  "Due Date": "",
  "Bid Submission Type": "",
  "Term of Bid": "",
  "Pre Bid Meeting": "",
  "Installation": "",
  "Bid Bond Requirement": "",
  "Delivery Date": "",
  "Payment Terms": "",
  "Any Additional Documentation Required": "",
  "MFG for Registration": "",
  "Contract or Cooperative to use": "",
  "Model_no": "",
  "Part_no": "",
  "Product": "",
  "contact_info": {
    "name": "",
    "phone": "",
    "email": ""
  },
  "company_name": "",
  "Bid Summary": "",
  "Product Specification": ""
}
```

If any field is not present, the system fills it with `"N/A"` or a contextual inference.

---

### **3. Quick Extraction (LLM Only)**

* For one-time use, choose **Quick Extract**.
* The document is processed **without FAISS indexing**, directly generating JSON output via **Google Gemini**.
* Ideal for smaller or individual RFP documents.

---

## 📁 Output Format

All extracted JSONs are saved inside the `outputs/` folder:

```
outputs/
├─ Dell_Laptops_Extended_Warranty.json
├─ Student_Devices_RFP.json
└─ Sample_RFP.json
```

Each output JSON follows a consistent, strict schema like this:

```json
{
  "Bid Number": "BPM044557",
  "Title": "Dell Laptops w/Extended Warranty",
  "Due Date": "06/10/2024 02:00 PM EDT",
  "Bid Submission Type": "RFP - Request for Proposal (Informal)",
  "Term of Bid": "N/A",
  "Pre Bid Meeting": "N/A",
  "Installation": "N/A",
  "Bid Bond Requirement": "N/A",
  "Delivery Date": "06/10/2024",
  "Payment Terms": "N/A",
  "Any Additional Documentation Required": "N/A",
  "MFG for Registration": "Dell",
  "Contract or Cooperative to use": "No",
  "Model_no": "5550, WD22TB4",
  "Part_no": "CC7802",
  "Product": "Dell Latitude 5550, Dell Thunderbolt 4 Dock – WD22TB4",
  "contact_info": {
    "name": "Tamaira Hawkins",
    "phone": "410-260-7533",
    "email": "Thawkins@treasurer.state.md.us"
  },
  "company_name": "State of Maryland Treasurer's Office",
  "Bid Summary": "This solicitation is for Dell Laptops w/Extended Warranty.",
  "Product Specification": "Laptops must be Microsoft Copilot ready"
}
```

---

## 🧾 Evaluation Readiness

This project is designed to meet all **evaluation criteria**:

| Criterion              | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **Accuracy**           | Extracts well-structured JSON matching field names exactly                         |
| **Robustness**         | Handles both PDF and HTML RFP formats; adaptable schema                            |
| **Code Quality**       | Modular, clean, and fully documented codebase                                      |
| **Use of LLM/NLP/RAG** | Integrates Google Gemini (LLM) and FAISS (RAG) efficiently for scalable processing |

---

## 🌟 Summary

✅ Supports **both structured LLM + RAG extraction** and **Quick LLM-only extraction**.
✅ Maintains **consistent JSON schema** for every RFP processed.
✅ Ready for **future extension** — Q&A, retrieval, and analytics can easily be added using the FAISS index.
✅ Simple Streamlit UI for evaluators to upload and evaluate RFPs interactively.

---
