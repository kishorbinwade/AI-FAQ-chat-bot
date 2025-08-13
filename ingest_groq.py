from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Folder with your documents
DATA_PATH = "data"
VECTOR_DB_PATH = "faiss_index"

def load_documents():
    """Load all documents from the data directory"""
    docs = []
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory '{DATA_PATH}' not found!")
        print(f"Please create the directory and add your documents (.pdf, .docx, .txt)")
        return docs
    
    files = os.listdir(DATA_PATH)
    if not files:
        print(f"⚠️ No files found in '{DATA_PATH}' directory!")
        return docs
    
    print(f"📂 Loading documents from '{DATA_PATH}'...")
    
    for file in files:
        file_path = os.path.join(DATA_PATH, file)
        try:
            if file.endswith(".pdf"):
                print(f"  📄 Loading PDF: {file}")
                docs.extend(PyPDFLoader(file_path).load())
            elif file.endswith(".docx"):
                print(f"  📝 Loading DOCX: {file}")
                docs.extend(Docx2txtLoader(file_path).load())
            elif file.endswith(".txt"):
                print(f"  📃 Loading TXT: {file}")
                docs.extend(TextLoader(file_path, encoding='utf-8').load())
            else:
                print(f"  ⏭️ Skipping unsupported file: {file}")
        except Exception as e:
            print(f"  ❌ Error loading {file}: {str(e)}")
    
    print(f"✅ Loaded {len(docs)} document chunks")
    return docs

def ingest():
    """Main ingestion function"""
    print("🚀 Starting document ingestion process...")
    
    # Load documents
    docs = load_documents()
    if not docs:
        print("❌ No documents to process. Exiting.")
        return
    
    print(f"📊 Processing {len(docs)} documents...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")
    
    # Initialize embeddings
    print("🔧 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    print("🗄️ Creating FAISS vector database...")
    db = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Save the vector store
    print(f"💾 Saving FAISS index to '{VECTOR_DB_PATH}'...")
    db.save_local(VECTOR_DB_PATH)
    
    print(f"✅ FAISS index successfully saved to '{VECTOR_DB_PATH}'")
    print("🎉 Document ingestion completed!")
    print("\nYou can now:")
    print("1. Start your FastAPI server: uvicorn main:app --reload")
    print("2. Open your chatbot at: http://127.0.0.1:8000")

def create_sample_data():
    """Create sample data directory and files for testing"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
        # Create sample FAQ file
        sample_content = """Company FAQ

Q: What are our office hours?
A: Our office hours are Monday to Friday, 9 AM to 6 PM.

Q: How do I request vacation time?
A: Submit a vacation request through the HR portal at least 2 weeks in advance.

Q: What is our company's return policy?
A: We offer a 30-day return policy for all products with original receipt.

Q: How do I contact technical support?
A: You can reach technical support at support@company.com or call 1-800-SUPPORT.

Q: What benefits do employees receive?
A: Employees receive health insurance, dental coverage, 401k matching, and paid time off.
"""
        
        with open(os.path.join(DATA_PATH, "company_faq.txt"), "w", encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"✅ Created sample data in '{DATA_PATH}/company_faq.txt'")
        print("Add your own documents to this folder before running ingestion.")

if __name__ == "__main__":
    # Check if data directory exists, create sample if not
    if not os.path.exists(DATA_PATH):
        print(f"📁 Data directory '{DATA_PATH}' not found.")
        create_sample = input("Would you like to create a sample data directory? (y/n): ").lower().strip()
        if create_sample in ['y', 'yes']:
            create_sample_data()
        else:
            print("Please create the data directory and add your documents, then run this script again.")
            exit()
    
    ingest()