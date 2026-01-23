import React, { useState } from 'react';

export default function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [architectureDiagram, setArchitectureDiagram] = useState(null);
  const [dataFlowDiagram, setDataFlowDiagram] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('analyze');

  const handleSearch = async () => {
    const res = await fetch('http://127.0.0.1:5000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query }),
    });
    const data = await res.json();
    setResult(data);
  };

  const loadArchitectureDiagram = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5000/api/model-architecture');
      const data = await res.json();
      setArchitectureDiagram(data.diagram);
    } catch (error) {
      console.error('Failed to load architecture diagram:', error);
    }
  };

  const loadDataFlowDiagram = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5000/api/data-flow');
      const data = await res.json();
      setDataFlowDiagram(data.diagram);
    } catch (error) {
      console.error('Failed to load data flow diagram:', error);
    }
  };

  const loadModelInfo = async () => {
    try {
      const res = await fetch('http://127.0.0.1:5000/api/model-info');
      const data = await res.json();
      setModelInfo(data);
    } catch (error) {
      console.error('Failed to load model info:', error);
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
        <button
          onClick={() => {setActiveTab('architecture'); loadArchitectureDiagram(); loadModelInfo();}}
          className={`px-6 py-3 font-medium ${activeTab === 'architecture' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
        >
          🏗️ Model Architecture
        </button>
        <button
          onClick={() => {setActiveTab('dataflow'); loadDataFlowDiagram();}}
          className={`px-6 py-3 font-medium ${activeTab === 'dataflow' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
        >
          🔄 Data Flow
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
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-medium"
          >
            🔍 Analyze with DRAGON
          </button>

          {result && (
            <div className="mt-8">
              <h2 className="text-2xl font-semibold mb-4">🐉 DRAGON Analysis Results:</h2>
              
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

      {/* Architecture Tab */}
      {activeTab === 'architecture' && (
        <div>
          <h2 className="text-2xl font-semibold mb-6">🏗️ DRAGON Model Architecture</h2>
          
          {modelInfo && (
            <div className="mb-8 p-6 border rounded-lg bg-gray-50">
              <h3 className="text-xl font-semibold mb-4">📋 Model Specifications</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Architecture Details</h4>
                  <ul className="text-sm space-y-1">
                    <li><strong>Type:</strong> {modelInfo.architecture.type}</li>
                    <li><strong>Layers:</strong> {modelInfo.architecture.layers}</li>
                    <li><strong>Hidden Size:</strong> {modelInfo.architecture.hidden_size}</li>
                    <li><strong>Attention Heads:</strong> {modelInfo.architecture.attention_heads}</li>
                    <li><strong>Vocabulary Size:</strong> {modelInfo.architecture.vocabulary_size}</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Capabilities</h4>
                  <ul className="text-sm space-y-1">
                    {modelInfo.capabilities.map((cap, i) => (
                      <li key={i}>• {cap}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {architectureDiagram && (
            <div className="border rounded-lg p-4 bg-white">
              <img 
                src={`data:image/png;base64,${architectureDiagram}`} 
                alt="DRAGON Model Architecture" 
                className="w-full h-auto"
              />
            </div>
          )}
          
          {!architectureDiagram && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="mt-2 text-gray-600">Loading architecture diagram...</p>
            </div>
          )}
        </div>
      )}

      {/* Data Flow Tab */}
      {activeTab === 'dataflow' && (
        <div>
          <h2 className="text-2xl font-semibold mb-6">🔄 DRAGON Data Flow</h2>
          
          {dataFlowDiagram && (
            <div className="border rounded-lg p-4 bg-white">
              <img 
                src={`data:image/png;base64,${dataFlowDiagram}`} 
                alt="DRAGON Data Flow" 
                className="w-full h-auto"
              />
            </div>
          )}
          
          {!dataFlowDiagram && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="mt-2 text-gray-600">Loading data flow diagram...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
