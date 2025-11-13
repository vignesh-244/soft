# Document Chatbot Backend

A FastAPI backend service for the Document Chatbot application that provides AI-powered chat functionality for uploaded documents.

## Features

- **Chat API**: Handle chat messages and generate intelligent responses about documents
- **AI Integration**: Uses OpenAI GPT models for natural language responses (with fallback to rule-based responses)
- **Supabase Integration**: Stores chat sessions, messages, and document metadata
- **CORS Support**: Configured for frontend communication
- **Health Checks**: API health monitoring endpoint

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   - Copy `.env.example` to `.env`
   - Fill in your Supabase and OpenAI credentials:
     ```
     SUPABASE_URL=your_supabase_project_url
     SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
     OPENAI_API_KEY=your_openai_api_key  # Optional
     ```

3. **Run the Server**:
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### POST `/api/chat`
Send a chat message and receive an AI-generated response.

**Request Body**:
```json
{
  "session_id": "uuid-of-chat-session",
  "message": "What is this document about?",
  "document_id": "uuid-of-document"  // Optional if session already has document
}
```

**Response**:
```json
{
  "message": "AI-generated response about the document",
  "session_id": "uuid-of-chat-session"
}
```

### GET `/api/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-13T15:30:00.000Z"
}
```

## Architecture

- **FastAPI**: Web framework for building APIs
- **Supabase**: Database and real-time features
- **OpenAI**: AI language model for chat responses
- **Pydantic**: Data validation and serialization

## Response Generation

The backend supports two response generation methods:

1. **AI-Powered Responses** (when OpenAI API key is provided):
   - Uses GPT-3.5-turbo model
   - Context-aware responses based on document content
   - Natural language processing

2. **Rule-Based Responses** (fallback):
   - Keyword matching
   - Document statistics
   - Simple pattern recognition

## Security

- Uses Supabase service role key for backend operations
- Validates user access to chat sessions and documents
- Row Level Security (RLS) enforced through Supabase policies

## Development

- **Auto-reload**: Server restarts automatically on code changes
- **CORS**: Configured for local development (ports 5173, 3000)
- **Error Handling**: Comprehensive error responses and logging
