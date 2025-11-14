# Document Chatbot Backend

A FastAPI backend service for the Document Chatbot application that provides AI-powered chat functionality for uploaded documents.
## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   - Copy `.env.example` to `.env`
     ```
     GEMINI_API_KEY=your_openai_api_key 
     ```

3. **Run the Server**:
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
