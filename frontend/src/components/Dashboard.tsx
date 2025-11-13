import { useState } from 'react';
import { LogOut } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { Document } from '../lib/supabase';
import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';
import ChatInterface from './ChatInterface';

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadComplete = () => {
    setRefreshTrigger((prev) => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <img src="/images.jpeg" alt="Cogentsoft" className="h-10" />
            <div className="border-l border-slate-300 pl-4">
              <h1 className="text-xl font-bold text-slate-800">Document Chatbot</h1>
              <p className="text-xs text-slate-500">Powered by Cogentsoft</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {user ? (
              <>
                <div className="text-right">
                  <p className="text-sm font-medium text-slate-700">{user.email}</p>
                  <p className="text-xs text-slate-500">Signed in</p>
                </div>
                <button
                  onClick={signOut}
                  className="p-2 text-slate-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition"
                  title="Sign out"
                >
                  <LogOut size={20} />
                </button>
              </>
            ) : (
              <div className="text-right">
                <p className="text-sm font-medium text-slate-700">Demo Mode</p>
                <p className="text-xs text-slate-500">Not signed in</p>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
      

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <DocumentUpload onUploadComplete={handleUploadComplete} />
            <DocumentList
              onSelectDocument={setSelectedDocument}
              selectedDocumentId={selectedDocument?.id}
              refreshTrigger={refreshTrigger}
            />
          </div>

          <div className="lg:col-span-2">
            <ChatInterface selectedDocument={selectedDocument} />
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-slate-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          <p>© {new Date().getFullYear()} Cogentsoft. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
