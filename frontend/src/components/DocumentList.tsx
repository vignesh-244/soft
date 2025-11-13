import { useEffect, useState } from 'react';
import { FileText, Trash2, Calendar, HardDrive } from 'lucide-react';
import { Document } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';

interface DocumentListProps {
  onSelectDocument: (doc: Document) => void;
  selectedDocumentId?: string;
  refreshTrigger: number;
}



export default function DocumentList({
  onSelectDocument,
  selectedDocumentId,
  refreshTrigger,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    if (user) {
      loadDocuments();
    }
  }, [user, refreshTrigger]);

  const loadDocuments = async () => {
    setLoading(true);

    // Get uploaded documents from localStorage
    const uploadedDocs = JSON.parse(localStorage.getItem('uploaded-documents') || '[]');

    setDocuments(uploadedDocs);
    setLoading(false);
  };

  const deleteDocument = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this document?')) return;

    // For demo purposes, just show a message
    alert('Delete functionality is disabled in demo mode');
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <p className="text-slate-500 text-center">Loading documents...</p>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
        <FileText className="mx-auto text-slate-300 mb-3" size={48} />
        <p className="text-slate-600 font-medium">No documents yet</p>
        <p className="text-slate-500 text-sm mt-1">Upload a document to get started</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="p-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Your Documents</h2>
        <p className="text-sm text-slate-500 mt-1">{documents.length} documents</p>
      </div>
      <div className="divide-y divide-slate-200 max-h-96 overflow-y-auto">
        {documents.map((doc) => (
          <div
            key={doc.id}
            onClick={() => onSelectDocument(doc)}
            className={`p-4 cursor-pointer transition hover:bg-slate-50 ${
              selectedDocumentId === doc.id ? 'bg-orange-50' : ''
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-3 flex-1 min-w-0">
                <div
                  className={`p-2 rounded-lg flex-shrink-0 ${
                    selectedDocumentId === doc.id ? 'bg-orange-200' : 'bg-slate-100'
                  }`}
                >
                  <FileText
                    size={18}
                    className={selectedDocumentId === doc.id ? 'text-orange-700' : 'text-slate-600'}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-slate-800 truncate">{doc.title}</h3>
                  <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      <Calendar size={12} />
                      {formatDate(doc.upload_date)}
                    </span>
                    <span className="flex items-center gap-1">
                      <HardDrive size={12} />
                      {formatFileSize(doc.file_size)}
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={(e) => deleteDocument(doc.id, e)}
                className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition flex-shrink-0"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
