import React from 'react';
import { Activity, FileText, UploadCloud, History } from 'lucide-react';

interface HeaderProps {
  activeTab: 'new' | 'report' | 'index';
  setActiveTab: (tab: 'new' | 'report' | 'index') => void;
  hasCurrentReport: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  hasCurrentReport,
}) => {
  return (
    <header className="swiss-header no-print">
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
          <Activity size={20} color="#15181B" strokeWidth={2.5} />
          <h1 style={{ fontSize: '20px', letterSpacing: '-0.02em', margin: 0 }}>
            Stance Health
          </h1>
          <span style={{ color: 'var(--muted)', fontSize: '13px', borderLeft: '1px solid var(--line)', paddingLeft: '10px' }}>
            Clinical Assessment Form Pipeline
          </span>
        </div>
        <p style={{ fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Sports & Musculoskeletal Physiotherapy · Bengaluru
        </p>
      </div>

      <nav className="nav-tabs" style={{ display: 'flex', gap: '8px' }}>
        <button
          className={`btn ${activeTab === 'new' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('new')}
        >
          <UploadCloud size={16} />
          New Assessment
        </button>

        <button
          className={`btn ${activeTab === 'report' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('report')}
          disabled={!hasCurrentReport}
          style={{ opacity: hasCurrentReport ? 1 : 0.5 }}
        >
          <FileText size={16} />
          Current Report
        </button>

        <button
          className={`btn ${activeTab === 'index' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('index')}
        >
          <History size={16} />
          History
        </button>
      </nav>
    </header>
  );
};
