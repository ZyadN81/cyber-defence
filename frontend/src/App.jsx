import React, { useState } from "react";

const API_BASE = "/api";

function normalizeResults(data) {
  if (Array.isArray(data?.results)) {
    return data.results;
  }

  if (Array.isArray(data?.recommendations)) {
    return data.recommendations.map((r) => ({
      technique: r.tactic || r.technique || "Unknown Technique",
      confidence: r.confidence || "N/A",
      description: "Recommended by semantic similarity with D3FEND abstracts.",
      category: "D3FEND",
    }));
  }

  return [];
}

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("analyze");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");

  const requestAnalysis = async (payload) => {
    const endpoints = [`${API_BASE}/analyze`];
    let lastError = null;

    for (const endpoint of endpoints) {
      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        return await res.json();
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error("Failed to reach analysis endpoint.");
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError("");
    setResult(null);

    try {
      const data = await requestAnalysis({ problem: query, query });
      const normalized = normalizeResults(data);
      setResult({ ...data, results: normalized });
    } catch (error) {
      setAnalysisError(
        `Analysis failed. Check backend at ${API_BASE}. Details: ${error.message}`,
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="analysis-card">
        <header className="hero-header">
          <h1>🐉 DRAGON Cybersecurity Analysis System</h1>
          <p>
            Comprehensive cybersecurity assessment powered by neural networks
          </p>
        </header>

        <div className="tabs">
          <button
            onClick={() => setActiveTab("analyze")}
            className={`tab-btn ${activeTab === "analyze" ? "active" : ""}`}
          >
            🔍 Analysis
          </button>
        </div>

        {activeTab === "analyze" && (
          <section className="analysis-content">
            <h2 className="section-title">
              Cybersecurity Strategy Recommender
            </h2>

            <div className="controls">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                rows={5}
                placeholder="Describe your cybersecurity incident or problem... (e.g., 'Network Segmentation', 'Malware Detection', 'Intrusion Prevention')"
              />
              <button
                onClick={handleSearch}
                disabled={isAnalyzing || !query.trim()}
                type="button"
              >
                {isAnalyzing ? "⏳ Analyzing..." : "🔍 Analyze with DRAGON"}
              </button>
            </div>

            {isAnalyzing && (
              <p className="status-text">
                Model is processing your request. This can take a few seconds.
              </p>
            )}

            {analysisError && <p className="error-banner">{analysisError}</p>}

            {result && (
              <section className="results-block">
                <h2 className="section-title">🐉 DRAGON Analysis Results</h2>

                {Array.isArray(result.matches) && result.matches.length > 0 && (
                  <div className="results-panel">
                    <h3 className="panel-title">📚 Matched Abstracts</h3>
                    <div className="abstracts-grid">
                      {result.matches.map((m, i) => (
                        <article
                          key={`${m.id || "abs"}-${i}`}
                          className="abstract-card"
                        >
                          <div className="abstract-meta">
                            <span className="meta-chip">
                              ID: {m.id || "N/A"}
                            </span>
                            <span className="meta-chip blue">
                              Confidence: {m.confidence || "N/A"}
                            </span>
                          </div>
                          <p>{m.text || "No abstract text available."}</p>
                        </article>
                      ))}
                    </div>
                  </div>
                )}

                {result.analysis_summary && (
                  <div className="summary-grid">
                    <article className="summary-card">
                      <h3>📉 Confidence Score</h3>
                      <p className="summary-value">
                        {Number(result.analysis_summary.confidence_score ?? 0)}%
                      </p>
                      <div
                        className="progress-track"
                        role="progressbar"
                        aria-valuenow={Number(
                          result.analysis_summary.confidence_score ?? 0,
                        )}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      >
                        <span
                          className="progress-fill"
                          style={{
                            width: `${Math.max(0, Math.min(100, Number(result.analysis_summary.confidence_score ?? 0)))}%`,
                          }}
                        />
                      </div>
                      <p className="muted">Neural network certainty level</p>
                    </article>

                    <article className="summary-card">
                      <h3>⚠️ Risk Assessment</h3>
                      <span className="risk-pill">
                        {(
                          result.analysis_summary.risk_level || "High"
                        ).toUpperCase()}{" "}
                        RISK
                      </span>
                      <p className="muted">
                        AI-powered threat level evaluation
                      </p>
                    </article>

                    <article className="summary-card">
                      <h3>🏷️ Detected Labels</h3>
                      <div className="tags-wrap">
                        {(result.analysis_summary.threat_categories || []).map(
                          (label) => (
                            <span key={label} className="label-tag">
                              {label}
                            </span>
                          ),
                        )}
                      </div>
                      <p className="muted">
                        {
                          (result.analysis_summary.threat_categories || [])
                            .length
                        }{" "}
                        cybersecurity categories identified
                      </p>
                    </article>

                    <article className="summary-card method-card">
                      <h3>⚙️ Analysis Method</h3>
                      <p className="method-title">
                        {result.analysis_summary.method}
                      </p>
                      <p className="muted">
                        Processing time:{" "}
                        {result.analysis_summary.processing_time || "N/A"}s
                      </p>
                    </article>
                  </div>
                )}

                <div className="results-panel">
                  <h3 className="panel-title">
                    🛡️ Recommended D3FEND Techniques
                  </h3>
                  <div className="recommendation-list">
                    {result.results.map((r, i) => (
                      <article key={i} className="recommendation-card">
                        <div className="recommendation-head">
                          <h4>{r.technique}</h4>
                          <span className="meta-chip blue">{r.confidence}</span>
                        </div>
                        <p>{r.description}</p>
                        <span className="meta-chip">
                          Category: {r.category}
                        </span>
                      </article>
                    ))}
                  </div>

                  <aside className="integration-note">
                    <strong>D3FEND Integration:</strong> These tactics are
                    mapped from MITRE D3FEND knowledge base using neural network
                    embeddings for comprehensive defense strategy planning.
                  </aside>
                </div>
              </section>
            )}
          </section>
        )}
      </section>
    </main>
  );
}
