import React from 'react';
import { ShieldCheck, Code2 } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <header className="navbar">
      <div className="brand-wrapper">
        <div className="brand-logo-mark">
          <ShieldCheck size={18} strokeWidth={2.2} />
        </div>
        <div className="brand-name">
          <span>VeraVoice</span>
        </div>
      </div>

      <div className="nav-right">
        <div className="nav-sdk-link">
          <Code2 size={14} />
          <span>APIs & SDK — Coming Soon</span>
        </div>
      </div>
    </header>
  );
};
