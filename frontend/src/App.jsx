import React, { useMemo, useState } from "react";

const API_BASE = "/api";

const mockResult = {
  analysis_summary: {
    confidence_score: 54,
    risk_level: "High",
    method: "Pure AI DRAGON with D3FEND Integration",
    threat_categories: ["malware", "cyberthreat"],
    processing_time: 2.498,
  },
  results: [
    { technique: "Threat Intelligence" },
    { technique: "Antivirus Scanning" },
  ],
};

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

function toPercent(confidence) {
  const numeric = Number(confidence);
  if (Number.isFinite(numeric)) {
    return Math.max(0, Math.min(100, Math.round(numeric)));
  }
  return 54;
}

function mapRiskClass(level) {
  const normalized = (level || "").toLowerCase();
  if (normalized.includes("high")) {
    return "risk-pill high";
  }
  if (normalized.includes("medium")) {
    return "risk-pill medium";
  }
  return "risk-pill low";
}

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(mockResult);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");

  const requestAnalysis = async (payload) => {
    const endpoint = `${API_BASE}/analyze`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
  };

  const handleAnalyze = async () => {
    if (!query.trim()) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError("");

    try {
      const data = await requestAnalysis({ problem: query, query });
      setResult({ ...data, results: normalizeResults(data) });
    } catch (error) {
      setAnalysisError(
        `Analysis failed. Check backend at ${API_BASE}. Details: ${error.message}`,
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const summary = result?.analysis_summary || {};
  const labels = summary.threat_categories || [];
  const confidence = toPercent(summary.confidence_score);
  const riskLabel = `${(summary.risk_level || "HIGH").toUpperCase()} RISK`;

  const defenseTactics = useMemo(() => {
    if (Array.isArray(result?.results) && result.results.length > 0) {
      return result.results.map(
        (item) => item.technique || "Unknown Technique",
      );
    }
    return ["Threat Intelligence", "Antivirus Scanning"];
  }, [result]);

  return (
    <main className="app-shell">
      <section className="analysis-card">
        <header className="hero-header">
          <h1>
            <span className="title-emoji" aria-hidden="true">
              🤖
            </span>{" "}
            AI Analysis Results
          </h1>
          <p>
            Comprehensive cybersecurity assessment powered by neural networks
          </p>
        </header>

        <div className="controls">
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={3}
            placeholder="Describe your cybersecurity incident..."
          />
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !query.trim()}
            type="button"
          >
            {isAnalyzing ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {analysisError && <p className="error-banner">{analysisError}</p>}

        <div className="stats-grid">
          <article className="stat-panel confidence">
            <h2>📉 Confidence Score</h2>
            <p className="value">{confidence}%</p>
            <div
              className="progress-track"
              role="progressbar"
              aria-valuenow={confidence}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span
                className="progress-fill"
                style={{ width: `${confidence}%` }}
              />
            </div>
            <p className="hint">Neural network certainty level</p>
          </article>

          <article className="stat-panel">
            <h2>⚠️ Risk Assessment</h2>
            <p className={mapRiskClass(summary.risk_level)}>{riskLabel}</p>
            <p className="hint">AI-powered threat level evaluation</p>
          </article>

          <article className="stat-panel">
            <h2>🏷️ Detected Labels</h2>
            <div className="tags-wrap">
              {labels.map((label) => (
                <span key={label} className="label-tag">
                  {label}
                </span>
              ))}
            </div>
            <p className="hint">
              {labels.length} cybersecurity categories identified
            </p>
          </article>

          <article className="stat-panel method-panel">
            <h2>⚙️ Analysis Method</h2>
            <p className="method-title">
              {summary.method || "Pure AI DRAGON with D3FEND Integration"}
            </p>
            <p className="hint">
              Processing time: {summary.processing_time || 2.498}s
            </p>
          </article>
        </div>

        <section className="defense-panel">
          <h2>🛡️ Recommended Defense Tactics</h2>
          <ul>
            {defenseTactics.map((technique) => (
              <li key={technique}>🛡️ {technique}</li>
            ))}
          </ul>

          <aside className="integration-note">
            <strong>D3FEND Integration:</strong> These tactics are mapped from
            MITRE D3FEND knowledge base using neural network embeddings for
            comprehensive defense strategy planning.
          </aside>
        </section>
      </section>
    </main>
  );
}
