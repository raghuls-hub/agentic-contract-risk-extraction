import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Scale, ShieldAlert } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [fullText, setFullText] = useState("");
  const [pdfBase64, setPdfBase64] = useState("");
  const [selectedClauseId, setSelectedClauseId] = useState(null);
  const [activeTab, setActiveTab] = useState("pdf"); // 'pdf' or 'text'
  
  const textContainerRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const runAnalysis = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResults(null);
    setFullText("");
    setPdfBase64("");
    setSelectedClauseId(null);
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }
      
      setFullText(data.full_text);
      setPdfBase64(data.pdf_base64);
      setResults(data.results || []);
      
      if (data.results && data.results.length > 0) {
        setSelectedClauseId(data.results[0].chunk_id || "Clause 1");
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Scroll to clause when selected from list
  useEffect(() => {
    if (selectedClauseId && activeTab === 'text' && textContainerRef.current) {
      const markEl = textContainerRef.current.querySelector(`mark[data-id="${selectedClauseId}"]`);
      if (markEl) {
        markEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [selectedClauseId, activeTab]);

  const handleTextClick = (e) => {
    if (e.target.tagName === 'MARK') {
      const cid = e.target.getAttribute('data-id');
      if (cid) setSelectedClauseId(cid);
    }
  };

  const getHighlightedHtml = () => {
    if (!fullText) return { __html: "" };
    
    let html = fullText.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    if (results) {
      results.forEach((r, idx) => {
        const cid = r.chunk_id || `Clause ${idx+1}`;
        const ctext = r.clause_causing_risk || "";
        
        if (ctext) {
          const safeCtext = ctext.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
          const isSelected = selectedClauseId === cid;
          const className = isSelected ? "active" : "";
          const replacement = `<mark data-id="${cid}" class="${className}">${safeCtext}</mark>`;
          html = html.replace(safeCtext, replacement);
        }
      });
    }
    
    return { __html: html };
  };

  const selectedResult = results ? results.find((r, i) => (r.chunk_id || `Clause ${i+1}`) === selectedClauseId) : null;

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>⚖️ M&A Risk extraction</h1>
        <p>Agentic Contract Risk Analysis Dashboard</p>
      </header>

      {!results && (
        <div className="uploader-card">
          <FileText size={48} color="var(--accent-color)" style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ marginBottom: '1rem' }}>Upload M&A Contract</h2>
          <label className="upload-label">
            <input type="file" className="file-input" accept=".pdf" onChange={handleFileChange} />
            {file ? file.name : "Select PDF File"}
          </label>
          <br />
          <button 
            className="run-btn" 
            onClick={runAnalysis} 
            disabled={!file || loading}
          >
            {loading ? "Analyzing Document..." : "Run AI Analysis"}
          </button>
          {error && <p style={{ color: 'var(--danger-color)', marginTop: '1rem' }}>{error}</p>}
        </div>
      )}

      {results && (
        <div className="grid-container">
          
          {/* Left Column: List */}
          <div className="glass-panel">
            <div className="panel-header">
              <ShieldAlert size={20} /> Risk Clauses
            </div>
            
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ padding: '0.75rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid var(--success-color)', borderRadius: '0.5rem', color: 'var(--success-color)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle size={18} /> Analysis Complete! {results.length} found.
              </div>
            </div>

            <div className="clauses-list">
              {results.length === 0 ? (
                <p style={{ color: 'var(--text-secondary)' }}>No risk clauses found above threshold.</p>
              ) : (
                results.map((r, i) => {
                  const cid = r.chunk_id || `Clause ${i+1}`;
                  return (
                    <div 
                      key={cid} 
                      className={`clause-item ${selectedClauseId === cid ? 'selected' : ''}`}
                      onClick={() => setSelectedClauseId(cid)}
                    >
                      <div className="clause-id">{cid}</div>
                      <div className="clause-summary">{r.clause_summary || "Unknown Risk"}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Center Column: Document Viewer */}
          <div className="glass-panel">
            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'pdf' ? 'active' : ''}`} 
                onClick={() => setActiveTab('pdf')}
              >
                Original PDF
              </button>
              <button 
                className={`tab ${activeTab === 'text' ? 'active' : ''}`} 
                onClick={() => setActiveTab('text')}
              >
                Interactive Text
              </button>
            </div>
            
            {activeTab === 'pdf' ? (
              <div style={{ flex: 1, borderRadius: '0.5rem', overflow: 'hidden' }}>
                {pdfBase64 ? (
                  <iframe 
                    src={`data:application/pdf;base64,${pdfBase64}#toolbar=0`} 
                    width="100%" 
                    height="100%" 
                    style={{ border: 'none' }}
                    title="PDF Viewer"
                  />
                ) : (
                  <p>No PDF available</p>
                )}
              </div>
            ) : (
              <div 
                className="doc-content" 
                ref={textContainerRef}
                onClick={handleTextClick}
                dangerouslySetInnerHTML={getHighlightedHtml()}
              />
            )}
          </div>

          {/* Right Column: Scorecard */}
          <div className="glass-panel">
            <div className="panel-header">
              <Scale size={20} /> Analysis & Reasoning
            </div>
            
            {selectedResult ? (
              <div className="scorecard-content">
                <div className="stat-box">
                  <div className="stat-title">Dominant Party</div>
                  <div className="stat-value" style={{ color: 'var(--accent-color)' }}>
                    {selectedResult.dominant_party || "Unknown"}
                  </div>
                </div>

                {selectedResult.mathematical_balance && (
                  <div className="stat-box">
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                      <div>
                        <div className="stat-title">Score A</div>
                        <div className="stat-value">{Number(selectedResult.mathematical_balance.score_company_a || 0).toFixed(1)}/100</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div className="stat-title">Score B</div>
                        <div className="stat-value">{Number(selectedResult.mathematical_balance.score_company_b || 0).toFixed(1)}/100</div>
                      </div>
                    </div>
                    
                    <div className="progress-container">
                      <div className="progress-label">
                        <span>Company A (Buyer)</span>
                        <span>{selectedResult.mathematical_balance.company_a_favorability_pct || 0}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill" 
                          style={{ width: `${selectedResult.mathematical_balance.company_a_favorability_pct || 0}%` }}
                        ></div>
                      </div>

                      <div className="progress-label">
                        <span>Company B (Target)</span>
                        <span>{selectedResult.mathematical_balance.company_b_favorability_pct || 0}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill target" 
                          style={{ width: `${selectedResult.mathematical_balance.company_b_favorability_pct || 0}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      <strong>Delta:</strong> {selectedResult.mathematical_balance.negotiation_delta_pct || 0} pts
                    </div>
                  </div>
                )}

                <div className="info-box reason">
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <AlertTriangle size={16} /> Legal Reason
                  </strong>
                  {selectedResult.reason_for_risk || "No reasoning provided."}
                </div>

                <div className="info-box compromise">
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <Scale size={16} /> Suggested Compromise
                  </strong>
                  {selectedResult.suggested_compromise || "No compromise provided."}
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Select a clause to view details.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
