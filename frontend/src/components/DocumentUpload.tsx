import { useState } from 'react';
import { Upload, FileText, X, CheckCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface DocumentUploadProps {
  onUploadComplete: () => void;
}

export default function DocumentUpload({ onUploadComplete }: DocumentUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const { user } = useAuth();

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !user) return;

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      // Read file as ArrayBuffer for binary files like PDFs
      const arrayBuffer = await file.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);

      // Convert to base64
      let binary = '';
      uint8Array.forEach(byte => binary += String.fromCharCode(byte));
      const base64Content = btoa(binary);

      // Upload to backend API
      const response = await fetch('http://localhost:8000/api/documents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: file.name,
          content: base64Content,
          file_type: file.type || 'text/plain',
          file_size: file.size,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload document');
      }

      const uploadedDoc = await response.json();

      // Also store in localStorage for frontend display
      const localDoc = {
        id: uploadedDoc.id,
        user_id: user.id,
        title: file.name,
        content: base64Content, // Store base64 content for local display/fallback
        file_type: file.type || 'application/pdf',
        file_size: file.size,
        upload_date: uploadedDoc.upload_date,
        created_at: uploadedDoc.created_at,
      };

      // Get existing uploaded documents
      const existingUploads = JSON.parse(localStorage.getItem('uploaded-documents') || '[]');

      // Add new document
      existingUploads.push(localDoc);

      // Save back to localStorage
      localStorage.setItem('uploaded-documents', JSON.stringify(existingUploads));

      setSuccess(`Successfully uploaded "${file.name}"`);
      onUploadComplete();
      e.target.value = '';

      // Clear success message after 3 seconds
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-orange-100 rounded-lg">
          <FileText className="text-orange-600" size={20} />
        </div>
        <h2 className="text-lg font-semibold text-slate-800">Upload Document</h2>
      </div>

      <div className="relative">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          disabled={uploading}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer transition ${
            uploading
              ? 'border-slate-300 bg-slate-50 cursor-not-allowed'
              : 'border-slate-300 hover:border-orange-400 hover:bg-orange-50'
          }`}
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <Upload className="mb-2 text-slate-400" size={32} />
            <p className="text-sm text-slate-600 font-medium">
              {uploading ? 'Uploading...' : 'Click to upload document'}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              PDF files only
            </p>
          </div>
        </label>
      </div>

      {success && (
        <div className="mt-4 bg-green-50 text-green-700 px-4 py-3 rounded-lg text-sm flex items-start gap-2">
          <CheckCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span>{success}</span>
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-50 text-red-700 px-4 py-3 rounded-lg text-sm flex items-start gap-2">
          <X size={16} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
