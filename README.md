# 🤖 RAG Chatbot — PDF Question Answering System

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions about them in natural language. Built with a fully local stack — no API costs, no data sent to the cloud.

---

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🔍 Semantic search over document content using FAISS vector index
- 🧠 Local LLM inference via Ollama — fully offline, no API key needed
- ⚡ Sub-3-second end-to-end response time
- 🖥️ Clean Streamlit web interface

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Text Extraction & Chunking
    ↓
Embedding Generation (HuggingFace)
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval (Top-K chunks)
    ↓
Ollama LLM (local inference)
    ↓
Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Ollama (local) |
| Vector Store | FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| PDF Parsing | PyMuPDF / pdfplumber |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### Installation

```bash
# Clone the repository
git clone https://github.com/sakshibhutekar/rag-chatbot.git
cd rag-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Requirements.txt
```

### Pull the Ollama model

```bash
ollama pull llama3
```

### Run the app

```bash
streamlit run App.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
rag-chatbot/
├── App.py                  # Streamlit UI and app entry point
├── rag_core.py             # Core RAG pipeline (chunking, retrieval, generation)
├── pdf_upload_handler.py   # PDF ingestion and text extraction
├── Requirements.txt        # Python dependencies
└── .gitignore
```

---

## 📊 Performance

- ⚡ Response time: **< 3 seconds** end-to-end
- 🗂️ Supports multi-page, multi-document PDFs
- 🔒 Fully local — no data leaves your machine

---

## 🔮 Future Improvements

- [ ] Add chat history / multi-turn conversation support
- [ ] Support for more file types (DOCX, TXT, CSV)
- [ ] Swap FAISS for ChromaDB for persistent storage
- [ ] Add source highlighting in retrieved chunks
- [ ] Docker containerization

---

## 👩‍💻 Author

**Sakshi Bhutekar**
- GitHub: [@sakshibhutekar](https://github.com/sakshibhutekar)
- Email: sakshibhtekar.work@gmail.com
