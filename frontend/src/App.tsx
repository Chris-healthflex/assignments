import React, { useState } from 'react';
import { Header } from './components/Header';
import { NewAssessment } from './components/NewAssessment';
import { ReportView } from './components/ReportView';
import { IndexView } from './components/IndexView';
import { FirstAssessment, ConfidenceReport, ParseResponseMeta, AssessmentRecord } from './types/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'new' | 'report' | 'index'>('new');
  const [currentAssessment, setCurrentAssessment] = useState<FirstAssessment | null>(null);
  const [currentConfidence, setCurrentConfidence] = useState<ConfidenceReport | null>(null);
  const [currentMeta, setCurrentMeta] = useState<ParseResponseMeta | undefined>(undefined);
  const [currentRecordId, setCurrentRecordId] = useState<string | undefined>(undefined);

  const handleAssessmentParsed = (assessment: FirstAssessment, meta: ParseResponseMeta) => {
    setCurrentAssessment(assessment);
    setCurrentMeta(meta);
    setCurrentRecordId(undefined);
    setActiveTab('report');
  };

  const handleSelectRecord = (record: AssessmentRecord) => {
    setCurrentAssessment(record.assessment);
    setCurrentConfidence(record.confidence || null);
    setCurrentRecordId(record.id);
    setCurrentMeta(undefined);
    setActiveTab('report');
  };

  return (
    <div className="swiss-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasCurrentReport={!!currentAssessment}
      />

      <main>
        {activeTab === 'new' && (
          <NewAssessment onAssessmentParsed={handleAssessmentParsed} />
        )}

        {activeTab === 'report' && currentAssessment && (
          <ReportView
            assessment={currentAssessment}
            confidence={currentConfidence}
            meta={currentMeta}
            recordId={currentRecordId}
            onSaved={(id) => setCurrentRecordId(id)}
          />
        )}

        {activeTab === 'index' && (
          <IndexView onSelectRecord={handleSelectRecord} />
        )}
      </main>
    </div>
  );
};
