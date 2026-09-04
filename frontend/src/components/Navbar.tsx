import { ShieldCheck, Server, ExternalLink } from 'lucide-react';
import { getApiBaseUrl } from '../utils/apiConfig';

export const Navbar = () => {
  const apiDocsUrl = `${getApiBaseUrl()}/docs`;

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

      <div className="nav-right" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <a
          href={apiDocsUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="nav-sdk-link"
          style={{ textDecoration: 'none', color: 'inherit' }}
        >
          <Server size={14} />
          <span>API Docs</span>
          <ExternalLink size={12} style={{ opacity: 0.6 }} />
        </a>
      </div>
    </header>
  );
};
