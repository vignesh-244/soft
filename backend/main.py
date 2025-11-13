import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=gemini_api_key)

# Application settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIR))

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# In-memory storage for chat messages (in production, use a database)
chat_messages: Dict[str, List[Dict[str, Any]]] = {}

# FastAPI app
app = FastAPI(title="Document Chatbot API", version="1.0.0")

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentUploadRequest(BaseModel):
    title: str
    content: str  # base64 encoded
    file_type: str
    file_size: int

class DocumentResponse(BaseModel):
    id: str
    title: str
    file_type: str
    file_size: int
    upload_date: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_id: str

class ChatResponse(BaseModel):
    message: str

class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str

# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def embed_chunks(chunks: List[str]) -> Any:
    """Generate embeddings for text chunks."""
    return embedding_model.encode(chunks)

def store_in_chromadb(chunks: List[str], embeddings: Any, collection_name: str):
    """Store chunks and embeddings in ChromaDB."""
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add documents to collection
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def query_chromadb(query: str, collection, n_results: int = 3) -> str:
    """Query ChromaDB for relevant chunks."""
    query_embedding = embedding_model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return "\n".join(results['documents'][0])

def generate_response(query: str, context: str, conversation_history: List[Dict[str, Any]] = None) -> str:
    """Generate response using Gemini AI with conversation history."""
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    # Build conversation context (last 10 messages)
    conversation_context = ""
    if conversation_history:
        recent_messages = conversation_history[-10:]  # Last 10 messages
        conversation_lines = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_lines.append(f"{role}: {msg['content']}")
        conversation_context = "\n".join(conversation_lines)

    prompt = f"""You are a helpful AI assistant that answers questions about documents. You have access to relevant document context and conversation history.

Document Context:
{context}

Conversation History:
{conversation_context}

Current User Query: {query}

Instructions:
- Use the document context to provide accurate information
- Consider the conversation history for context and follow-up questions
- If the context doesn't contain enough information, say so politely
- Provide clear, well-formatted responses
- Use bullet points and bold text for structured information

Answer:"""

    response = gemini_model.generate_content(prompt)
    return response.text

# API endpoints
@app.post("/api/documents", response_model=DocumentResponse)
async def upload_document(request: DocumentUploadRequest):
    """Upload and process a PDF document."""
    import base64

    if not request.title.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Check file size
    if request.file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")

    # Decode base64 content
    try:
        file_content = base64.b64decode(request.content)
    except:
        raise HTTPException(status_code=400, detail="Invalid base64 content")

    # Save file temporarily
    temp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(file_content)

    try:
        # Extract text
        text = extract_text_from_pdf(temp_path)

        # Split into chunks
        chunks = split_text(text)

        # Generate embeddings
        embeddings = embed_chunks(chunks)

        # Store in ChromaDB
        doc_id = str(uuid.uuid4())
        collection_name = f"doc_{doc_id}"
        store_in_chromadb(chunks, embeddings, collection_name)

        # Create response
        now = datetime.utcnow().isoformat()
        response = DocumentResponse(
            id=doc_id,
            title=request.title,
            file_type=request.file_type,
            file_size=request.file_size,
            upload_date=now,
            created_at=now
        )

        return response

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document information (placeholder - in production, store metadata)."""
    # For now, just return basic info
    # In a real app, you'd store document metadata in a database
    return {"id": doc_id, "content": "Document content stored in ChromaDB"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Chat with a document using RAG."""
    collection_name = f"doc_{request.document_id}"

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get relevant context
    context = query_chromadb(request.message, collection)

    # Get conversation history (excluding current message)
    conversation_history = chat_messages.get(request.session_id, [])[:-1] if request.session_id in chat_messages else []

    # Generate response with conversation history
    response_text = generate_response(request.message, context, conversation_history)

    # Store messages
    if request.session_id not in chat_messages:
        chat_messages[request.session_id] = []

    # Add user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "session_id": request.session_id,
        "role": "user",
        "content": request.message,
        "created_at": datetime.utcnow().isoformat()
    }
    chat_messages[request.session_id].append(user_msg)

    # Add assistant message
    assistant_msg = {
        "id": str(uuid.uuid4()),
        "session_id": request.session_id,
        "role": "assistant",
        "content": response_text,
        "created_at": datetime.utcnow().isoformat()
    }
    chat_messages[request.session_id].append(assistant_msg)

    return ChatResponse(message=response_text)

@app.get("/api/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(session_id: str):
    """Get chat messages for a session."""
    messages = chat_messages.get(session_id, [])
    return [MessageResponse(**msg) for msg in messages]

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
