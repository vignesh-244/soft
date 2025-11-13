import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, FileText } from 'lucide-react';
import { Document } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';

interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

interface ChatInterfaceProps {
  selectedDocument: Document | null;
}

export default function ChatInterface({ selectedDocument }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  useEffect(() => {
    if (selectedDocument) {
      // Create a session ID based on document and user (or demo user)
      const userId = user?.id || 'demo-user';
      const sessionId = `session-${userId}-${selectedDocument.id}`;
      setCurrentSessionId(sessionId);
      loadMessages(sessionId);
    } else {
      setCurrentSessionId(null);
      setMessages([]);
    }
  }, [selectedDocument, user]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadMessages = async (sessionId: string) => {
    try {
      // Try to load from backend API
      const response = await fetch(`http://localhost:8000/api/messages/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
        return;
      }
    } catch (error) {
      console.warn('Backend not available, using local storage:', error);
    }

    // Fallback to localStorage
    const stored = localStorage.getItem(`chat-${sessionId}`);
    if (stored) {
      setMessages(JSON.parse(stored));
    }
  };

  const getDocumentContent = async (docId: string): Promise<string> => {
    // First check if we have the document content locally
    const localDocs = JSON.parse(localStorage.getItem('uploaded-documents') || '[]');
    const localDoc = localDocs.find((doc: any) => doc.id === docId);
    if (localDoc) {
      return localDoc.content;
    }

    // Try to fetch from backend
    try {
      const response = await fetch(`http://localhost:8000/api/documents/${docId}`);
      if (response.ok) {
        const docData = await response.json();
        return docData.content;
      }
    } catch (error) {
      console.warn('Could not fetch document content from backend:', error);
    }

    // Return empty string if not found
    return '';
  };

  const handleSend = async () => {
    if (!input.trim() || !currentSessionId || !selectedDocument) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message immediately
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      session_id: currentSessionId,
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString()
    };

    const updatedMessagesWithUser = [...messages, userMsg];
    setMessages(updatedMessagesWithUser);
    setLoading(true);

    try {
      // Get the document content for local fallback
      const docContent = await getDocumentContent(selectedDocument.id);

      // Call backend API for chat response
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: currentSessionId,
          message: userMessage,
          document_id: selectedDocument.id,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get response from backend');
      }

      const data = await response.json();

      // Add the assistant message
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        session_id: currentSessionId,
        role: 'assistant',
        content: data.message,
        created_at: new Date().toISOString()
      };

      const updatedMessages = [...updatedMessagesWithUser, assistantMsg];
      setMessages(updatedMessages);

      // Also save to localStorage as backup
      localStorage.setItem(`chat-${currentSessionId}`, JSON.stringify(updatedMessages));
    } catch (error) {
      console.error('Error sending message:', error);
      // Fallback to local response generation if backend fails
      const docWithContent = { ...selectedDocument, content: await getDocumentContent(selectedDocument.id) };
      const assistantResponse = generateResponse(userMessage, docWithContent);

      // Add assistant message
      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        session_id: currentSessionId,
        role: 'assistant',
        content: assistantResponse,
        created_at: new Date().toISOString()
      };

      const updatedMessages = [...updatedMessagesWithUser, assistantMsg];
      setMessages(updatedMessages);

      // Save to localStorage
      localStorage.setItem(`chat-${currentSessionId}`, JSON.stringify(updatedMessages));
    }

    setLoading(false);
  };

  const generateResponse = (question: string, doc: Document): string => {
    const content = doc.content.toLowerCase();
    const query = question.toLowerCase();

    if (query.includes('what') && query.includes('about')) {
      return `This document "${doc.title}" contains ${doc.content.length} characters of text. It appears to be a ${doc.file_type} file uploaded on ${new Date(doc.upload_date).toLocaleDateString()}.`;
    }

    if (query.includes('summary') || query.includes('summarize')) {
      const preview = doc.content.substring(0, 300);
      return `Here's a brief overview of the document: ${preview}${doc.content.length > 300 ? '...' : ''}`;
    }

    if (query.includes('how many')) {
      const words = doc.content.split(/\s+/).length;
      const lines = doc.content.split('\n').length;
      return `The document contains approximately ${words} words and ${lines} lines.`;
    }

    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 0);
    const relevantSentences = sentences
      .filter((sentence) => {
        const words = query.split(' ').filter((w) => w.length > 3);
        return words.some((word) => sentence.includes(word));
      })
      .slice(0, 2);

    if (relevantSentences.length > 0) {
      return `Based on the document content: ${relevantSentences.join('. ')}.`;
    }

    return `I've analyzed the document "${doc.title}". Could you please be more specific about what information you're looking for? You can ask about the content, request a summary, or ask specific questions about the document.`;
  };

  if (!selectedDocument) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-[600px] flex items-center justify-center">
        <div className="text-center">
          <FileText className="mx-auto text-slate-300 mb-3" size={64} />
          <h3 className="text-lg font-semibold text-slate-700 mb-2">
            No Document Selected
          </h3>
          <p className="text-slate-500">
            Select a document from the list to start chatting
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-[600px] flex flex-col">
      <div className="p-4 border-b border-slate-200 bg-gradient-to-r from-orange-50 to-white">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-100 rounded-lg">
            <FileText className="text-orange-600" size={20} />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">{selectedDocument.title}</h3>
            <p className="text-sm text-slate-500">Ask questions about this document</p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Bot className="mx-auto text-slate-300 mb-3" size={48} />
            <p className="text-slate-600 font-medium mb-2">Start a conversation</p>
            <p className="text-slate-500 text-sm">
              Ask questions about the document and I'll help you find answers
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {message.role === 'assistant' && (
              <div className="p-2 bg-orange-100 rounded-lg h-fit">
                <Bot size={18} className="text-orange-600" />
              </div>
            )}
            <div
              className={`max-w-[70%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-orange-500 text-white'
                  : 'bg-slate-100 text-slate-800'
              }`}
            >
              <p className="text-sm leading-relaxed">{message.content}</p>
            </div>
            {message.role === 'user' && (
              <div className="p-2 bg-slate-700 rounded-lg h-fit">
                <User size={18} className="text-white" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3 justify-start">
            <div className="p-2 bg-orange-100 rounded-lg h-fit">
              <Bot size={18} className="text-orange-600" />
            </div>
            <div className="bg-slate-100 rounded-lg p-3 max-w-[70%]">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-orange-500 border-t-transparent"></div>
                <span className="text-sm text-slate-600">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-slate-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter pressed, calling handleSend');
                console.log('Input:', input.trim());
                console.log('Session ID:', currentSessionId);
                console.log('Selected Document:', selectedDocument?.id);
                handleSend();
              }
            }}
            placeholder="Ask a question about the document..."
            disabled={loading}
            className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none transition disabled:bg-slate-50"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
