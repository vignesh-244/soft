from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from openai import OpenAI
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
import logging
import asyncio
from contextlib import asynccontextmanager
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import io
import base64
import pdfplumber
import google.generativeai as genai

# Unstructured imports
from unstructured.partition.auto import partition
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.staging.base import elements_to_json

class DocumentExtractor:
    """Handles extraction of text from various document formats using Unstructured"""

    @staticmethod
    def extract_text_with_unstructured(content: str, file_type: str, filename: str = "document") -> str:
        """
        Extract text from various document formats using Unstructured library

        Supported formats:
        - PDF (both text-based and scanned with OCR)
        - Word documents (.docx)
        - PowerPoint (.pptx)
        - HTML
        - Plain text
        - And many more formats
        """
        try:
            # Convert content to bytes
            if content.startswith('%PDF') or content.startswith('PK'):  # PDF or Office formats
                try:
                    file_bytes = content.encode('utf-8')
                except UnicodeEncodeError:
                    file_bytes = content.encode('latin-1', errors='ignore')
            elif content.startswith('JVBERi0'):  # Base64 encoded
                file_bytes = base64.b64decode(content)
            else:
                # Try base64 decode first
                try:
                    file_bytes = base64.b64decode(content)
                except Exception:
                    # If not base64, treat as plain text
                    if file_type in ["text/plain", "text/markdown"]:
                        return content
                    file_bytes = content.encode('latin-1', errors='ignore')

            # Create a BytesIO object
            file_io = io.BytesIO(file_bytes)

            # Only PDF files are supported now
            if file_type == "application/pdf" or content.startswith('%PDF'):
                logger.info("Processing PDF with multiple extractors...")
                extractors_tried = []

                # Try pdfplumber first
                try:
                    file_io.seek(0)
                    with pdfplumber.open(file_io) as pdf:
                        text_pages = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_pages.append(page_text.strip())
                        extracted_text = "\n\n".join(text_pages)

                    if extracted_text.strip():
                        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF using pdfplumber")
                        return extracted_text
                    else:
                        extractors_tried.append("pdfplumber (no text)")
                except Exception as pdf_error:
                    logger.warning(f"pdfplumber failed: {pdf_error}")
                    extractors_tried.append(f"pdfplumber ({str(pdf_error)})")

                # Try PyMuPDF (fitz)
                try:
                    import fitz
                    file_io.seek(0)
                    doc = fitz.open(stream=file_io.read(), filetype="pdf")
                    text_pages = []
                    for page in doc:
                        page_text = page.get_text()
                        if page_text and page_text.strip():
                            text_pages.append(page_text.strip())
                    extracted_text = "\n\n".join(text_pages)

                    if extracted_text.strip():
                        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF using PyMuPDF")
                        return extracted_text
                    else:
                        extractors_tried.append("PyMuPDF (no text)")
                except Exception as fitz_error:
                    logger.warning(f"PyMuPDF failed: {fitz_error}")
                    extractors_tried.append(f"PyMuPDF ({str(fitz_error)})")

                # Try unstructured as last resort
                try:
                    file_io.seek(0)
                    elements = partition(file=file_io)
                    extracted_text = "\n\n".join([str(element) for element in elements])
                    if extracted_text.strip():
                        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF using unstructured")
                        return extracted_text
                    else:
                        extractors_tried.append("unstructured (no text)")
                except Exception as unstructured_error:
                    logger.warning(f"unstructured failed: {unstructured_error}")
                    extractors_tried.append(f"unstructured ({str(unstructured_error)})")

                    # Check if it's a Poppler error (OCR needed)
                    error_msg = str(unstructured_error).lower()
                    if "poppler" in error_msg or "page count" in error_msg:
                        raise HTTPException(
                            status_code=400,
                            detail="This PDF appears to be scanned/image-based and requires OCR. "
                                   "Please install Poppler and Tesseract-OCR: "
                                   "Ubuntu: 'sudo apt-get install poppler-utils tesseract-ocr' | "
                                   "macOS: 'brew install poppler tesseract' | "
                                   "Windows: Install from github.com/oschwartz10612/poppler-windows"
                        )

                # Check if PDFs appear to be corrupted
                corruption_indicators = ["zlib error", "corrupted data", "object out of range", "xref size"]
                has_corruption = any(any(indicator in error.lower() for indicator in corruption_indicators) for error in extractors_tried)

                if has_corruption:
                    raise HTTPException(
                        status_code=400,
                        detail="This PDF file appears to be corrupted or damaged. "
                               "Please try with a different PDF file or repair the current one. "
                               f"Technical details: {', '.join(extractors_tried)}"
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to extract text from PDF. This may be a scanned/image-based PDF. "
                               f"Attempted extractors: {', '.join(extractors_tried)}. "
                               f"For scanned PDFs, install Poppler and Tesseract-OCR."
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported. Please upload a PDF document."
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error extracting text with unstructured: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from document: {str(e)}"
            )

    @staticmethod
    def extract_text(content: str, file_type: str, filename: str = "document") -> str:
        """Extract text based on file type using Unstructured"""
        return DocumentExtractor.extract_text_with_unstructured(content, file_type, filename)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyAkAlU_8yPcY1f2A7OJ1Tasx4BeE7X313o")
    MAX_FILE_SIZE: int = 10_485_760  # 10MB
    MAX_DOCUMENTS: int = 1000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient
    CHROMA_PERSIST_DIR: str = "./chroma_db"

settings = Settings()

# Initialize Sentence Transformer for embeddings
class EmbeddingService:
    """Handles embeddings using Sentence Transformers"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise Exception("Embedding model not initialized")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise Exception("Embedding model not initialized")
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# Initialize embedding service
embedding_service = EmbeddingService()

# ChromaDB Vector Store
class ChromaVectorStore:
    """Vector store using ChromaDB"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"description": "Document chunks for RAG"}
            )
            
            logger.info(f"ChromaDB initialized. Collection size: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, doc_id: str, doc_title: str, chunks: List[str]):
        """Add document chunks to ChromaDB"""
        try:
            # Generate embeddings
            embeddings = embedding_service.embed_documents(chunks)
            
            # Create IDs for each chunk
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_id": i,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks for document '{doc_title}' to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    async def search(self, query: str, doc_id: Optional[str] = None, top_k: int = 3) -> List[Dict]:
        """Search for similar chunks in ChromaDB"""
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed_text(query)
            
            # Build where filter
            where_filter = {"doc_id": doc_id} if doc_id else None
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get ChromaDB statistics"""
        try:
            count = self.collection.count()
            # Get unique document IDs
            all_metadata = self.collection.get(include=["metadatas"])
            unique_docs = len(set(m["doc_id"] for m in all_metadata["metadatas"])) if all_metadata["metadatas"] else 0
            
            return {
                "total_chunks": count,
                "total_documents": unique_docs,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_chunks": 0, "total_documents": 0}
    
    async def delete_document(self, doc_id: str):
        """Delete all chunks for a document"""
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")

# Initialize ChromaDB
chroma_store = ChromaVectorStore()

# Document Processing Service
class DocumentProcessor:
    """Handles document chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def chunk_document(self, text: str, doc_id: str, doc_title: str) -> List[str]:
        """Split document into chunks"""
        try:
            # Ensure we have content to chunk
            if not text or not text.strip():
                raise Exception("No text content to chunk")

            chunks = self.text_splitter.split_text(text.strip())

            # Ensure we have at least one chunk
            if not chunks:
                # If chunking produces no chunks, create a single chunk with the whole text
                chunks = [text.strip()]
                logger.warning(f"Chunking produced no chunks, using single chunk for '{doc_title}'")

            logger.info(f"Split document '{doc_title}' into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    
    async def process_and_store_document(self, doc_id: str, doc_title: str, content: str):
        """Process document and add to ChromaDB"""
        try:
            # Chunk the document
            chunks = self.chunk_document(content, doc_id, doc_title)
            
            # Add to ChromaDB (embeddings generated inside)
            await chroma_store.add_documents(doc_id, doc_title, chunks)
            
            logger.info(f"Successfully processed and stored document: {doc_title}")
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise HTTPException(status_code=500, detail="Failed to process document")

doc_processor = DocumentProcessor()

# AI Service for generating responses
class AIService:
    """Handles AI response generation using Google Gemini"""

    def __init__(self):
        # Initialize Gemini
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini AI model initialized")
        else:
            self.model = None
            logger.warning("No Gemini API key provided")

    async def retrieve_relevant_chunks(self, query: str, doc_id: Optional[str] = None, top_k: int = 3) -> List[str]:
        """Retrieve relevant document chunks using ChromaDB"""
        try:
            results = await chroma_store.search(query, doc_id, top_k)
            return [result["content"] for result in results]
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    async def generate_response(self, question: str, document: 'DocumentInfo') -> str:
        """Generate AI response using RAG with Gemini"""
        if not self.model:
            return self._generate_simple_response(question, document)

        try:
            # Retrieve relevant chunks from ChromaDB
            relevant_chunks = await self.retrieve_relevant_chunks(question, document.id, top_k=3)

            # Use chunks or full content as context
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                logger.info(f"Using {len(relevant_chunks)} relevant chunks for context")
            else:
                context = document.content[:4000]
                logger.info("Using truncated document content as context")

            prompt = f"""You are a helpful assistant that answers questions about documents.
Use the following context from the document "{document.title}" to answer the user's question.
If the context doesn't contain enough information to answer the question, say so politely.
Be concise but informative.

Context:
{context}

Question: {question}

Answer:"""

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_simple_response(question, document)
    
    def _generate_simple_response(self, question: str, document: 'DocumentInfo') -> str:
        """Fallback response generation without AI"""
        content = document.content.lower()
        query = question.lower()
        
        if 'what' in query and 'about' in query:
            return f"This document \"{document.title}\" contains {len(document.content)} characters of text."
        
        if 'summary' in query or 'summarize' in query:
            preview = document.content[:300]
            return f"Here's a brief overview: {preview}{'...' if len(document.content) > 300 else ''}"
        
        if 'how many' in query:
            words = len(document.content.split())
            return f"The document contains approximately {words} words."
        
        # Simple keyword matching
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        query_words = [w for w in query.split() if len(w) > 3]
        
        relevant_sentences = []
        for sentence in sentences:
            if any(word in sentence for word in query_words):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 2:
                    break
        
        if relevant_sentences:
            return f"Based on the document: {'. '.join(relevant_sentences)}."
        
        return f"I've analyzed \"{document.title}\". Could you be more specific about what you're looking for?"

ai_service = AIService()

# Data Models
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = Field(None, max_length=100)

class ChatResponse(BaseModel):
    message: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class DocumentInfo(BaseModel):
    id: str
    title: str
    content: str
    file_type: str

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    file_type: str
    file_size: int = Field(..., gt=0, le=settings.MAX_FILE_SIZE)

class DocumentResponse(BaseModel):
    id: str
    title: str
    file_type: str
    file_size: int
    upload_date: str
    created_at: str

# In-Memory Data Store (for documents, sessions, messages)
class DataStore:
    """In-memory storage for application data"""
    
    def __init__(self):
        self.documents: Dict[str, Dict] = {
            "sample-doc-1": {
                "id": "sample-doc-1",
                "title": "Sample AI Document",
                "content": """Artificial Intelligence (AI) is revolutionizing how we interact with technology. 
                Machine learning, a subset of AI, enables computers to learn from data without explicit programming.
                Deep learning uses neural networks with multiple layers to process complex patterns.
                Natural Language Processing (NLP) allows machines to understand and generate human language.
                Computer vision enables machines to interpret and analyze visual information from the world.
                AI applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis.
                The future of AI holds promise for solving complex problems in healthcare, climate change, and education.""",
                "file_type": "text/plain"
            }
        }
        self.sessions: Dict[str, Dict] = {}
        self.messages: Dict[str, List[Dict]] = {}
    
    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document by ID"""
        doc_data = self.documents.get(document_id)
        if doc_data:
            return DocumentInfo(**doc_data)
        return None
    
    async def add_document(self, doc_id: str, title: str, content: str, file_type: str) -> Dict:
        """Add new document"""
        if len(self.documents) >= settings.MAX_DOCUMENTS:
            raise HTTPException(status_code=400, detail="Maximum document limit reached")
        
        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "content": content,
            "file_type": file_type
        }
        return self.documents[doc_id]
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    async def get_session(self, session_id: str) -> Dict:
        """Get or create session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "user_id": "default-user",
                "document_id": "sample-doc-1",
                "title": "Chat Session",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        return self.sessions[session_id]
    
    async def save_message(self, session_id: str, role: str, content: str):
        """Save chat message"""
        if session_id not in self.messages:
            self.messages[session_id] = []
        
        self.messages[session_id].append({
            "id": f"msg-{len(self.messages[session_id])}",
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        })
    
    async def get_messages(self, session_id: str) -> List[Dict]:
        """Get messages for session"""
        return self.messages.get(session_id, [])
    
    async def update_session(self, session_id: str):
        """Update session timestamp"""
        if session_id in self.sessions:
            self.sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        return list(self.documents.values())

data_store = DataStore()

# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    logger.info(f"OpenAI API Key present: {bool(settings.OPENAI_API_KEY)}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info("Using Unstructured library for document extraction")
    
    # Initialize ChromaDB with existing documents
    logger.info("Initializing ChromaDB with sample documents...")
    for doc_id, doc_data in data_store.documents.items():
        try:
            await doc_processor.process_and_store_document(
                doc_id, 
                doc_data["title"], 
                doc_data["content"]
            )
        except Exception as e:
            logger.error(f"Error initializing document {doc_id}: {e}")
    
    stats = chroma_store.get_stats()
    logger.info(f"ChromaDB initialized: {stats}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="Document Chatbot API with Unstructured",
    version="4.0.0",
    description="RAG-based document chatbot using Unstructured, ChromaDB and Sentence Transformers",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000", 
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Handle chat messages and generate responses using RAG"""
    try:
        logger.info(f"Chat request received: session_id={request.session_id}, message='{request.message[:50]}...'")

        # Validate and get session
        session = await data_store.get_session(request.session_id)

        # Get document
        doc_id = request.document_id or session.get("document_id")
        if not doc_id:
            raise HTTPException(status_code=400, detail="No document specified")

        document = await data_store.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info(f"Processing chat for document: {document.title} (ID: {doc_id})")

        # Save user message
        await data_store.save_message(request.session_id, "user", request.message)

        # Generate AI response using RAG (ChromaDB + Sentence Transformers)
        assistant_response = await ai_service.generate_response(request.message, document)

        logger.info(f"Generated response: '{assistant_response[:100]}...'")

        # Save assistant message
        await data_store.save_message(request.session_id, "assistant", assistant_response)

        # Update session
        await data_store.update_session(request.session_id)

        return ChatResponse(
            message=assistant_response,
            session_id=request.session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/documents", response_model=DocumentResponse)
async def upload_document(doc: DocumentUpload):
    """Upload a new document and add to ChromaDB"""
    try:
        # Validate file type - only PDF files allowed
        if doc.file_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported. Please upload a PDF document."
            )

        # Validate file size
        if doc.file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )

        # Extract text from document using Unstructured
        logger.info(f"Extracting text from {doc.title} ({doc.file_type})")
        extracted_content = DocumentExtractor.extract_text(doc.content, doc.file_type, doc.title)
        logger.info(f"Extracted {len(extracted_content)} characters from {doc.title}")

        # Generate unique ID
        doc_id = f"doc-{int(datetime.utcnow().timestamp() * 1000)}"

        # Store document with extracted text
        await data_store.add_document(doc_id, doc.title, extracted_content, doc.file_type)

        # Process and add to ChromaDB
        await doc_processor.process_and_store_document(doc_id, doc.title, extracted_content)

        logger.info(f"Document uploaded successfully: {doc.title} (ID: {doc_id})")

        return DocumentResponse(
            id=doc_id,
            title=doc.title,
            file_type=doc.file_type,
            file_size=doc.file_size,
            upload_date=datetime.utcnow().isoformat(),
            created_at=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    """Get all documents"""
    try:
        documents = data_store.get_all_documents()
        return [
            {
                "id": doc["id"],
                "title": doc["title"],
                "file_type": doc["file_type"],
                "file_size": len(doc["content"]),
                "upload_date": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat()
            }
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get specific document"""
    document = await data_store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from both data store and ChromaDB"""
    try:
        # Delete from ChromaDB
        await chroma_store.delete_document(document_id)
        
        # Delete from data store
        deleted = await data_store.delete_document(document_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    """Get messages for a chat session"""
    try:
        messages = await data_store.get_messages(session_id)
        return messages
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    chroma_stats = chroma_store.get_stats()
    return {
        "chromadb": chroma_stats,
        "documents": len(data_store.documents),
        "sessions": len(data_store.sessions),
        "embedding_model": settings.EMBEDDING_MODEL,
        "openai_enabled": bool(settings.OPENAI_API_KEY),
        "extractor": "Unstructured (pdfplumber for PDFs)"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "embedding_model": settings.EMBEDDING_MODEL,
        "extractor": "Unstructured Library"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Chatbot API with Unstructured, ChromaDB and Sentence Transformers",
        "version": "4.0.0",
        "extractor": "Unstructured",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
