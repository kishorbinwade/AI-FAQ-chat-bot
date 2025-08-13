# AI-FAQ-chat-bot

# 🤖 FAQ Chatbot with RAG Setup Guide

This is a complete setup guide for your enhanced FAQ Chatbot with Retrieval-Augmented Generation (RAG) capabilities.

## 📁 Project Structure
```
faq-chatbot/
├── main.py                 # Enhanced FastAPI server with RAG
├── ingest_groq.py         # Document ingestion script
├── index.html             # Enhanced frontend with RAG features
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/                  # Your documents folder (will be created)
│   ├── company_faq.txt    # Sample FAQ file
│   ├── your_docs.pdf      # Your PDF files
│   └── your_docs.docx     # Your DOCX files
└── faiss_index/           # Vector database (created after ingestion)
```

## 🚀 Quick Setup

### 1. **Create Project Directory**
```bash
mkdir faq-chatbot
cd faq-chatbot
```

### 2. **Download All Files**
Download all the artifacts provided:
- `main.py` - Enhanced RAG API
- `ingest_groq.py` - Document ingestion script  
- `index.html` - Enhanced frontend
- `requirements.txt` - Dependencies
- `.env.example` - Environment template

### 3. **Create Virtual Environment**
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 5. **Setup Environment Variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your key from: https://console.groq.com/keys
```

Edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 6. **Prepare Your Documents**
```bash
# Run the ingestion script (creates sample data if none exists)
python ingest_groq.py
```

This will:
- Create a `data/` folder
- Add sample FAQ content
- Process documents and create vector database
- Save to `faiss_index/` folder

### 7. **Start the API Server**
```bash
uvicorn main:app --reload
```

### 8. **Open the Chatbot**
- Open `index.html` in your web browser
- Or visit: `http://127.0.0.1:8000`

## 🎯 Features

### ✅ What's New:
- **RAG Integration**: Uses your documents to answer questions
- **Source Citations**: Shows which documents were referenced
- **Smart Fallback**: General knowledge when no docs match
- **Health Monitoring**: Check if knowledge base is loaded
- **Modern UI**: Improved design with typing indicators
- **File Support**: PDF, DOCX, and TXT files

### 📡 API Endpoints:
- `POST /ask` - RAG-powered Q&A with source citations
- `POST /chat` - General chat without RAG
- `GET /health` - System health and RAG status
- `POST /reload-knowledge-base` - Reload vector database
- `GET /` - API status

## 📝 Adding Your Documents

1. **Add files to the `data/` folder:**
   ```bash
   # Supported formats:
   data/company_policy.pdf
   data/employee_handbook.docx
   data/faq.txt
   ```

2. **Re-run ingestion:**
   ```bash
   python ingest_groq.py
   ```

3. **Reload knowledge base (optional):**
   ```bash
   curl -X POST http://127.0.0.1:8000/reload-knowledge-base
   ```

## 🔧 Customization

### Change LLM Model:
Edit `main.py`:
```python
llm = ChatGroq(
    model="llama3-8b-8192",  # Faster model
    groq_api_key=os.getenv("GROQ_API_KEY")
)
```

### Adjust Chunk Settings:
Edit `ingest_groq.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Smaller chunks
    chunk_overlap=100   # Less overlap
)
```

### Change Embeddings Model:
Edit both `main.py` and `ingest_groq.py`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

## 🐛 Troubleshooting

### Issue: "FAISS vector store not found"
**Solution:** Run `python ingest_groq.py` first

### Issue: "API Connection Failed"
**Solutions:**
- Check if server is running: `uvicorn main:app --reload`
- Verify API key in `.env` file
- Check Groq API key is valid

### Issue: "Error loading documents"
**Solutions:**
- Ensure files are in `data/` folder
- Check file formats (PDF, DOCX, TXT only)
- Verify file permissions
