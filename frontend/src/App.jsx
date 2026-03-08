import React, { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function normalizeResults(data) {
  if (Array.isArray(data?.results)) {
    return data.results;
  }

  if (Array.isArray(data?.recommendations)) {
    return data.recommendations.map((r) => ({
      technique: r.tactic || r.technique || 'Unknown Technique',
      confidence: r.confidence || 'N/A',
      description: 'Recommended by semantic similarity with D3FEND abstracts.',
      category: 'D3FEND',
    }));
  }

  return [];
}

export default function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('analyze');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState('');

  const requestAnalysis = async (payload) => {
    const endpoints = [`${API_BASE}/analyze`, `${API_BASE}/api/analyze`];
    let lastError = null;

    for (const endpoint of endpoints) {
      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
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

    throw lastError || new Error('Failed to reach analysis endpoint.');
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError('');
    setResult(null);

    try {
      const data = await requestAnalysis({ problem: query, query });
      const normalized = normalizeResults(data);
      setResult({ ...data, results: normalized });
    } catch (error) {
      setAnalysisError(
        `Analysis failed. Check backend at ${API_BASE}. Details: ${error.message}`
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-center">🐉 DRAGON Cybersecurity Analysis System</h1>
      
      {/* Navigation Tabs */}
      <div className="flex mb-6 border-b">
        <button
          onClick={() => setActiveTab('analyze')}
          className={`px-6 py-3 font-medium ${activeTab === 'analyze' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
        >
          🔍 Analysis
        </button>
      </div>

      {/* Analysis Tab */}
      {activeTab === 'analyze' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Cybersecurity Strategy Recommender</h2>
          <textarea
            className="w-full p-4 border border-gray-300 rounded-lg mb-4 resize-none"
            rows="5"
            placeholder="Describe your cybersecurity incident or problem... (e.g., 'Network Segmentation', 'Malware Detection', 'Intrusion Prevention')"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            onClick={handleSearch}
            disabled={isAnalyzing || !query.trim()}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze with DRAGON'}
          </button>

          {isAnalyzing && (
            <p className="mt-3 text-sm text-blue-700">Model is processing your request. This can take a few seconds.</p>
          )}

          {analysisError && (
            <div className="mt-4 p-3 rounded-lg bg-red-50 text-red-700 border border-red-200 text-sm">
              {analysisError}
            </div>
          )}

          {result && (
            <div className="mt-8">
              <h2 className="text-2xl font-semibold mb-4">🐉 DRAGON Analysis Results:</h2>
              {Array.isArray(result.matches) && result.matches.length > 0 && (
                <div className="mb-6 p-6 border rounded-lg bg-slate-50">
                  <h3 className="text-xl font-semibold mb-4">📚 Matched Abstracts</h3>
                  <div className="grid gap-3 max-h-96 overflow-auto pr-1">
                    {result.matches.map((m, i) => (
                      <div key={`${m.id || 'abs'}-${i}`} className="p-3 border rounded-lg bg-white">
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                          <span className="text-xs bg-slate-200 px-2 py-1 rounded">ID: {m.id || 'N/A'}</span>
                          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            Confidence: {m.confidence || 'N/A'}
                          </span>
                        </div>
                        <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{m.text || 'No abstract text available.'}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {result.analysis_summary && (
                <div className="mb-6 p-6 border rounded-lg bg-blue-50">
                  <h3 className="font-semibold text-blue-800 text-lg mb-3">📊 Analysis Summary:</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <strong className="text-blue-700">Method:</strong>
                      <p className="text-sm">{result.analysis_summary.method}</p>
                    </div>
                    <div>
                      <strong className="text-blue-700">Risk Level:</strong>
                      <p className={`text-sm font-medium ${
                        result.analysis_summary.risk_level === 'High' ? 'text-red-600' :
                        result.analysis_summary.risk_level === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                      }`}>{result.analysis_summary.risk_level}</p>
                    </div>
                    <div>
                      <strong className="text-blue-700">Threat Categories:</strong>
                      <p className="text-sm">{result.analysis_summary.threat_categories.join(', ')}</p>
                    </div>
                  </div>
                </div>
              )}

              <h3 className="text-xl font-semibold mb-4">🛡️ Recommended D3FEND Techniques:</h3>
              <div className="grid gap-4">
                {result.results.map((r, i) => (
                  <div key={i} className="p-4 border rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-bold text-lg text-gray-800">{r.technique}</h4>
                      <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                        {r.confidence}
                      </span>
                    </div>
                    <p className="text-gray-600 mb-2">{r.description}</p>
                    <span className="inline-block bg-gray-200 text-gray-700 px-2 py-1 rounded text-xs">
                      Category: {r.category}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
