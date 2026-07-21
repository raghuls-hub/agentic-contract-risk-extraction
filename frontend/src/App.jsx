import React, { useState, useEffect, useRef } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Scale, ShieldAlert } from 'lucide-react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [pdfBase64, setPdfBase64] = useState("");
  const [selectedClauseId, setSelectedClauseId] = useState(null);

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
    setPdfBase64("");
    setSelectedClauseId(null);
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch("https://agentic-contract-risk-extraction.onrender.com/api/analyze", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || "Analysis failed");
      }
      
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


  const selectedResult = results ? results.find((r, i) => (r.chunk_id || `Clause ${i+1}`) === selectedClauseId) : null;

  const calculateOverallBalance = () => {
    if (!results || results.length === 0) return null;
    let totalA = 0;
    let totalB = 0;
    let count = 0;
    
    results.forEach(r => {
      if (r.mathematical_balance) {
        totalA += (r.mathematical_balance.company_a_favorability_pct || 0);
        totalB += (r.mathematical_balance.company_b_favorability_pct || 0);
        count++;
      }
    });
    
    if (count === 0) return null;
    return {
      a: (totalA / count).toFixed(1),
      b: (totalB / count).toFixed(1)
    };
  };

  const overallBalance = calculateOverallBalance();

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>⚖️ Agentic Contract Risk Extraction</h1>
        <p>Agentic Contract Risk Analysis Dashboard</p>
      </header>

      {!results && (
        <div className="uploader-card">
          <FileText size={48} color="var(--accent-color)" style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ marginBottom: '1rem' }}>Upload Contract</h2>
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
              <div style={{ padding: '0.75rem', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid var(--success-color)', borderRadius: '0.5rem', color: 'var(--success-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <CheckCircle size={18} /> Analysis Complete! {results.length} found.
              </div>
              {overallBalance && (
                <div style={{ padding: '0.75rem', background: '#f8fafc', border: '1px solid var(--border-color)', borderRadius: '0.5rem' }}>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Overall Contract Weightage</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', fontWeight: 600 }}>
                    <span style={{ color: 'var(--accent-color)' }}>Buyer (A): {overallBalance.a}%</span>
                    <span style={{ color: 'var(--success-color)' }}>Target (B): {overallBalance.b}%</span>
                  </div>
                </div>
              )}
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
