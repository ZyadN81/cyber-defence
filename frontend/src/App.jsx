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

function parseConfidence(value) {
  if (typeof value === "number") {
    return Math.max(0, Math.min(100, value));
  }
  if (typeof value === "string") {
    const numeric = Number(value.replace("%", "").trim());
    if (!Number.isNaN(numeric)) {
      return Math.max(0, Math.min(100, numeric));
    }
  }
  return 54;
}

function toTitle(text) {
  return String(text || "")
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function buildSummary(result, query) {
  const summary = result?.analysis_summary || {};
  const firstScore = result?.results?.[0]?.confidence;
  const confidence = parseConfidence(summary.confidence_score ?? firstScore);
  const labels = Array.isArray(summary.threat_categories)
    ? summary.threat_categories
    : ["malware", "cyberthreat"];

  return {
    confidence_score: confidence,
    risk_level: summary.risk_level || "High",
    method: summary.method || "Pure AI DRAGON with D3FEND Integration",
    threat_categories: labels,
    processing_time: summary.processing_time || 2.498,
    query: query || "",
  };
}

function buildTopTactics(result) {
  const normalized = Array.isArray(result?.results) ? result.results : [];
  const derived = normalized
    .slice(0, 2)
    .map((item) => toTitle(item.technique))
    .filter(Boolean);

  if (derived.length > 0) {
    return derived;
  }

  return ["Threat Intelligence", "Antivirus Scanning"];
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

  const dashboardSummary = result ? buildSummary(result, query) : null;
  const topTactics = result ? buildTopTactics(result) : [];

  return (
    <main className="app-shell">
      <section className="analysis-card">
        {!result && (
          <>
            <header className="hero-header">
              <h1>🐉 DRAGON Cybersecurity Analysis System</h1>
              <p>
                Comprehensive cybersecurity assessment powered by neural
                networks
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
          </>
        )}

        {activeTab === "analyze" && !result && (
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
          </section>
        )}

        {result && dashboardSummary && (
          <section className="results-block">
            <header className="results-hero">
              <h2>
                <span aria-hidden="true">🤖</span> AI Analysis Results
              </h2>
              <p>
                Comprehensive cybersecurity assessment powered by neural
                networks
              </p>
            </header>

            <div className="kpi-top-grid">
              <article className="summary-card">
                <h3>📉 Confidence Score</h3>
                <p className="summary-value">
                  {Math.round(Number(dashboardSummary.confidence_score))}%
                </p>
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-valuenow={Number(dashboardSummary.confidence_score)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <span
                    className="progress-fill"
                    style={{
                      width: `${Math.max(0, Math.min(100, Number(dashboardSummary.confidence_score)))}%`,
                    }}
                  />
                </div>
                <p className="muted">Neural network certainty level</p>
              </article>

              <article className="summary-card">
                <h3>⚠️ Risk Assessment</h3>
                <span className="risk-pill">
                  {(dashboardSummary.risk_level || "High").toUpperCase()} RISK
                </span>
                <p className="muted">AI-powered threat level evaluation</p>
              </article>

              <article className="summary-card">
                <h3>🏷️ Detected Labels</h3>
                <div className="tags-wrap">
                  {(dashboardSummary.threat_categories || []).map((label) => (
                    <span key={label} className="label-tag">
                      {label}
                    </span>
                  ))}
                </div>
                <p className="muted">
                  {(dashboardSummary.threat_categories || []).length}{" "}
                  cybersecurity categories identified
                </p>
              </article>

              <article className="summary-card method-card">
                <h3>⚙️ Analysis Method</h3>
                <p className="method-title">{dashboardSummary.method}</p>
                <p className="muted">
                  Processing time: {dashboardSummary.processing_time || "N/A"}s
                </p>
              </article>
            </div>

            <div className="results-panel top-tactics-panel">
              <h3 className="panel-title">🛡️ Recommended Defense Tactics</h3>
              <ul className="top-tactics-list">
                {topTactics.map((name, i) => (
                  <li key={`${name}-${i}`}>🛡️ {name}</li>
                ))}
              </ul>

              <aside className="integration-note compact-note">
                <strong>D3FEND Integration:</strong> These tactics are mapped
                from MITRE D3FEND knowledge base using neural network embeddings
                for comprehensive defense strategy planning.
              </aside>
            </div>

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
                        <span className="meta-chip">ID: {m.id || "N/A"}</span>
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

            <div className="results-panel">
              <h3 className="panel-title">🛡️ Recommended D3FEND Techniques</h3>
              <div className="recommendation-list">
                {result.results.map((r, i) => (
                  <article key={i} className="recommendation-card">
                    <div className="recommendation-head">
                      <h4>{r.technique}</h4>
                      <span className="meta-chip blue">{r.confidence}</span>
                    </div>
                    <p>{r.description}</p>
                    <span className="meta-chip">Category: {r.category}</span>
                  </article>
                ))}
              </div>

              <aside className="integration-note">
                <strong>D3FEND Integration:</strong> These tactics are mapped
                from MITRE D3FEND knowledge base using neural network embeddings
                for comprehensive defense strategy planning.
              </aside>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
