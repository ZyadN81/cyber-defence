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
  return null;
}

function collectConfidenceValues(result) {
  if (!Array.isArray(result?.results)) {
    return [];
  }

  return result.results
    .map((item) => parseConfidence(item?.confidence))
    .filter((value) => Number.isFinite(value));
}

function deriveRiskLevel(summaryRisk, confidence) {
  if (summaryRisk) {
    return summaryRisk;
  }

  if (!Number.isFinite(confidence)) {
    return null;
  }

  if (confidence >= 70) {
    return "High";
  }
  if (confidence >= 40) {
    return "Medium";
  }
  return "Low";
}

function deriveThreatLabels(result, query) {
  const summaryLabels = result?.analysis_summary?.threat_categories;
  if (Array.isArray(summaryLabels) && summaryLabels.length > 0) {
    return summaryLabels;
  }

  const hintHits = result?.metadata?.query_hint_hits;
  if (Array.isArray(hintHits) && hintHits.length > 0) {
    return [...new Set(hintHits.map((hit) => String(hit).toLowerCase()))].slice(
      0,
      3,
    );
  }

  const derived = new Set();
  const q = String(query || "").toLowerCase();

  if (/(malware|virus|trojan|worm|ransomware)/.test(q)) {
    derived.add("malware");
  }
  if (/(threat|attack|intrusion|breach|compromise)/.test(q)) {
    derived.add("cyberthreat");
  }
  if (/(phish|credential|account)/.test(q)) {
    derived.add("identity");
  }

  if (derived.size > 0) {
    return Array.from(derived).slice(0, 3);
  }

  if (Array.isArray(result?.results)) {
    result.results.forEach((item) => {
      const name = String(item?.technique || "").toLowerCase();
      if (name.includes("threat")) {
        derived.add("cyberthreat");
      }
      if (name.includes("loss") || name.includes("asset")) {
        derived.add("data-protection");
      }
      if (name.includes("security operations") || name.includes("soc")) {
        derived.add("secops");
      }
    });
  }

  return Array.from(derived).slice(0, 3);
}

function deriveMethod(result) {
  const summaryMethod = result?.analysis_summary?.method;
  if (summaryMethod) {
    return summaryMethod;
  }

  const simpleMode = result?.metadata?.simple_mode;
  if (simpleMode === true) {
    return "Hybrid TF-IDF + D3FEND Mapping";
  }
  if (simpleMode === false) {
    return "Transformer Embeddings + D3FEND Mapping";
  }

  if (Array.isArray(result?.matches) && result.matches.length > 0) {
    return "Semantic Retrieval + D3FEND Mapping";
  }

  if (Array.isArray(result?.results) && result.results.length > 0) {
    return "DRAGON + D3FEND Mapping";
  }

  return null;
}

function deriveProcessingTime(result) {
  const summaryTime = result?.analysis_summary?.processing_time;
  if (summaryTime) {
    return summaryTime;
  }

  const metadataTime = result?.metadata?.processing_time;
  if (metadataTime) {
    return metadataTime;
  }

  return null;
}

function toTitle(text) {
  return String(text || "")
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function buildSummary(result, query) {
  const summary = result?.analysis_summary || {};
  const scoreFromSummary = parseConfidence(summary.confidence_score);
  const allScores = collectConfidenceValues(result);
  const avgScore =
    allScores.length > 0
      ? allScores.reduce((acc, value) => acc + value, 0) / allScores.length
      : null;
  const confidence = Number.isFinite(scoreFromSummary)
    ? scoreFromSummary
    : avgScore;
  const labels = deriveThreatLabels(result, query);
  const riskLevel = deriveRiskLevel(summary.risk_level, confidence);

  return {
    confidence_score: confidence,
    risk_level: riskLevel,
    method: deriveMethod(result),
    threat_categories: labels,
    processing_time: deriveProcessingTime(result),
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
    setResult((previous) => previous);

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
  const confidenceValue = Number(dashboardSummary?.confidence_score);
  const hasConfidence = Number.isFinite(confidenceValue);
  const confidencePercent = hasConfidence
    ? Math.max(0, Math.min(100, confidenceValue))
    : 0;

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
            {isAnalyzing && (
              <p className="status-text">
                Updating analysis results. This can take a few seconds.
              </p>
            )}

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
                  {hasConfidence ? `${Math.round(confidencePercent)}%` : "N/A"}
                </p>
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-valuenow={confidencePercent}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <span
                    className="progress-fill"
                    style={{
                      width: `${confidencePercent}%`,
                    }}
                  />
                </div>
                <p className="muted">Neural network certainty level</p>
              </article>

              <article className="summary-card">
                <h3>⚠️ Risk Assessment</h3>
                <span className="risk-pill">
                  {dashboardSummary.risk_level
                    ? `${dashboardSummary.risk_level.toUpperCase()} RISK`
                    : "N/A"}
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
                <p className="method-title">
                  {dashboardSummary.method || "N/A"}
                </p>
                <p className="muted">
                  Processing time:
                  {dashboardSummary.processing_time
                    ? ` ${dashboardSummary.processing_time}s`
                    : " N/A"}
                </p>
              </article>
            </div>

            <div className="results-panel top-tactics-panel">
              <h3 className="panel-title">🛡️ Recommended Defense Tactics</h3>
              <ul className="top-tactics-list">
                {topTactics.length > 0 ? (
                  topTactics.map((name, i) => (
                    <li key={`${name}-${i}`}>🛡️ {name}</li>
                  ))
                ) : (
                  <li>🛡️ N/A</li>
                )}
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
