# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
VECTOR_DB_PATH = "faiss_index"

# Initialize ChatGroq with API key
llm = ChatGroq(
    # model="llama-3.1-70b-versatile",
    
    model="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None

# Load vector store at startup
def load_vector_store():
    global vector_store
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS vector store loaded successfully")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            vector_store = None
    else:
        print("⚠️ Vector store not found. Run ingest_groq.py first to create the knowledge base.")

# Load vector store on startup
load_vector_store()

# Request model
class Question(BaseModel):
    question: str

# Response model
class Answer(BaseModel):
    answer: str
    sources: list = []

def get_relevant_context(query: str, k: int = 3):
    """Retrieve relevant documents from vector store"""
    if vector_store is None:
        return "", []
    
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(query, k=k)
        
        # Combine document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Extract source information
        sources = []
        for doc in docs:
            source_info = doc.metadata.get('source', 'Unknown')
            if source_info not in sources:
                sources.append(source_info)
        
        return context, sources
    
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "", []

def create_rag_prompt(question: str, context: str) -> str:
    """Create a prompt for RAG-based question answering"""
    return f"""You are a helpful assistant answering questions based on company documents and FAQ information.

Context Information:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context information
- If the answer is not found in the context, politely say you don't have that information
- Be concise and helpful
- Provide specific details when available in the context

Answer:"""

# POST endpoint for RAG-based QA
@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    try:
        # Get relevant context from vector store
        context, sources = get_relevant_context(q.question)
        
        if not context and vector_store is not None:
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or ask about topics covered in the company documents.",
                "sources": []
            }
        
        # Create messages for the LLM
        if context:
            # RAG-based response with context
            prompt = create_rag_prompt(q.question, context)
            messages = [
                SystemMessage(content="You are a helpful company FAQ assistant. Answer questions based on the provided context."),
                HumanMessage(content=prompt)
            ]
        else:
            # Fallback to general response if no vector store
            messages = [
                SystemMessage(content="You are a helpful assistant. Answer the user's question politely and helpfully."),
                HumanMessage(content=q.question)
            ]
        
        # Invoke the model
        response = llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": sources if context else []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# POST endpoint for general chat (without RAG)
@app.post("/chat")
async def chat(q: Question):
    """General chat endpoint without RAG"""
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=q.question)
        ]
        
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "FAQ Chatbot API is running", "rag_enabled": vector_store is not None}

# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_available": vector_store is not None,
        "vector_store_path": VECTOR_DB_PATH
    }

# Endpoint to reload vector store
@app.post("/reload-knowledge-base")
async def reload_knowledge_base():
    """Reload the vector store (useful after running ingest_groq.py)"""
    try:
        load_vector_store()
        return {
            "message": "Knowledge base reloaded successfully" if vector_store is not None else "Failed to load knowledge base",
            "rag_available": vector_store is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading knowledge base: {str(e)}")