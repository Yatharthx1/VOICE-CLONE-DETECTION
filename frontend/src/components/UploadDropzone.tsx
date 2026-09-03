import React, { useCallback, useRef, useState } from 'react';
import { Upload, FileAudio } from 'lucide-react';

interface UploadDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({ onFileSelected, disabled }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      onFileSelected(file);
    }
  }, [onFileSelected, disabled]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  }, [disabled]);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleClick = useCallback(() => {
    if (!disabled) inputRef.current?.click();
  }, [disabled]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelected(file);
      e.target.value = '';
    }
  }, [onFileSelected]);

  return (
    <>
      <div
        className={`dropzone-area ${isDragOver ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
      >
        <div className="dropzone-icon-box">
          {isDragOver ? <FileAudio size={22} /> : <Upload size={22} />}
        </div>
        <div className="dropzone-title">
          {isDragOver ? 'Drop audio file here' : 'Drop audio file or click to browse'}
        </div>
        <div className="dropzone-sub">
          WAV, MP3, M4A, FLAC, OGG — up to 50 MB
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
    </>
  );
};
