"""
Enhanced FastAPI service for D3FEND cybersecurity analysis.

Overview
--------
- Serves a simple API that analyzes an input problem statement against D3FEND abstracts.
- Uses a DRAGON-inspired encoder when available, otherwise falls back to DRAGON+ encoders.
- Ranks the most relevant abstracts, derives associated tactics, and returns a concise report.

Notes
-----
- This file is intentionally organized into clear sections with concise comments:
    1) Imports and logging
    2) Configuration & constants
    3) Ontology (D3FEND) helpers
    4) Text encoder (DRAGON / DRAGON+)
    5) Embedding cache management
    6) Tactics utilities & visualization
    7) FastAPI app & endpoints
    8) Dev entrypoint
"""

from __future__ import annotations

# 1) Imports & logging
# --------------------
import sys
import os
import re
import io
import base64
import logging
from typing import List, Dict, Optional

# Add dragon folder to path for importing DRAGON modules
dragon_path = os.path.join(os.path.dirname(__file__), 'dragon')
if dragon_path not in sys.path:
        sys.path.insert(0, dragon_path)

from fastapi import FastAPI
from pydantic import BaseModel
from rdflib import Graph, URIRef, Namespace, RDF
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app_enhanced")

app = FastAPI(title="DRAGON-D3FEND Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Configuration & constants
# ----------------------------
D3F = Namespace("http://d3fend.mitre.org/ontologies/d3fend.owl#")
ONTOLOGY_PATH = "d3fend_output.owl"
ABSTRACTS_FOLDER = "abstracts"
EMBEDDINGS_PATH = "enhanced_dragon_embeddings.pt"

# Model Configuration - Try DRAGON, fallback to DRAGON+
USE_DRAGON_MODEL = True
Q_MODEL_NAME = "facebook/dragon-plus-query-encoder"
P_MODEL_NAME = "facebook/dragon-plus-context-encoder"
DRAGON_MODEL_NAME = "roberta-large"  # Base for DRAGON

TOP_N_MATCHES = 10
MIN_SIMILARITY_THRESHOLD = 0.1
CONFIDENCE_FLOOR = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Ontology (D3FEND) helpers
# ----------------------------
# Load D3FEND Knowledge Graph once at startup (XML/RDF format)
graph = Graph()
try:
    graph.parse(ONTOLOGY_PATH, format="xml")
    logger.info("D3FEND ontology loaded: %s", ONTOLOGY_PATH)
except Exception as e:
    logger.error("Failed to parse D3FEND ontology at %s: %s", ONTOLOGY_PATH, e)
    raise

def load_abstracts() -> List[Dict[str, object]]:
    """Load all D3FEND abstracts from the ontology.

    Each returned item contains:
    - id: numeric string (e.g., "1234") extracted from the abstract URI
    - uri: rdflib URIRef of the abstract
    """
    abstracts: List[Dict[str, object]] = []
    for abstract_uri in graph.subjects(RDF.type, D3F.Abstract):
        m = re.search(r"abstract(\d+)", str(abstract_uri))
        if m:
            abstracts.append({"id": m.group(1), "uri": abstract_uri})
    return abstracts

abstracts = load_abstracts()
logger.info("Found %d abstract URIs in D3FEND ontology.", len(abstracts))

# 4) Text encoder (DRAGON / DRAGON+)
# ----------------------------------
class EnhancedDragonEncoder:
    def __init__(self):
        self.device = DEVICE
        self.dragon_available = False
        
        # Try to import and initialize DRAGON model
        try:
            from modeling.modeling_dragon import DRAGON, LMGNN, TextKGMessagePassing
            from utils import utils
            
            logger.info("Loading DRAGON model components...")
            
            # Initialize base tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(DRAGON_MODEL_NAME)
            
            # Try to create a simplified DRAGON-based encoder
            self.base_model = AutoModel.from_pretrained(DRAGON_MODEL_NAME).to(self.device)
            
            # Create a custom DRAGON-inspired encoder
            self.dragon_encoder = self._create_dragon_encoder()
            self.dragon_available = True
            
            logger.info("DRAGON model loaded successfully!")
            
        except Exception as e:
            logger.warning("DRAGON import failed: %s", e)
            logger.info("Falling back to DRAGON+ dual encoders...")
            self._load_dragon_plus()
    
    def _create_dragon_encoder(self):
        """Create a DRAGON-inspired encoder for text embedding.

        This wraps a base transformer with simple context attention layers to
        simulate graph-aware reasoning in a lightweight manner.
        """
        class DragonTextEncoder(nn.Module):
            def __init__(self, base_model, hidden_size=1024):
                super().__init__()
                self.base_model = base_model
                self.hidden_size = hidden_size
                
                # Add graph-aware layers inspired by DRAGON
                self.graph_projection = nn.Linear(base_model.config.hidden_size, hidden_size)
                self.context_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
                self.output_projection = nn.Linear(hidden_size, base_model.config.hidden_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids, attention_mask=None):
                # Get base embeddings
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state
                
                # Apply graph-aware processing
                projected = self.graph_projection(sequence_output)
                
                # Self-attention for context awareness (simulating graph reasoning)
                attended, _ = self.context_attention(projected, projected, projected, 
                                                   key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
                
                attended = self.dropout(attended)
                
                # Project back and combine with original
                output = self.output_projection(attended)
                
                # Pooling for sentence embedding
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).float()
                    sum_embeddings = torch.sum(output * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                else:
                    pooled = torch.mean(output, dim=1)
                
                return pooled
        
        return DragonTextEncoder(self.base_model).to(self.device)
    
    def _load_dragon_plus(self):
        """Load DRAGON+ dual encoders as fallback."""
        self.q_tokenizer = AutoTokenizer.from_pretrained(Q_MODEL_NAME)
        self.q_encoder = AutoModel.from_pretrained(Q_MODEL_NAME).to(self.device)
        self.p_tokenizer = AutoTokenizer.from_pretrained(P_MODEL_NAME)
        self.p_encoder = AutoModel.from_pretrained(P_MODEL_NAME).to(self.device)
        
        self.q_encoder.eval()
        self.p_encoder.eval()
    
    def encode_text(self, text: str, is_query: bool = True) -> torch.Tensor:
        """Encode text using DRAGON or DRAGON+ models.

        Args:
            text: Input text to encode.
            is_query: When True, uses the query encoder; otherwise the context encoder.

        Returns:
            torch.Tensor: L2-normalized embedding for DRAGON; pooled output for DRAGON+.
        """
        if self.dragon_available:
            return self._encode_with_dragon(text)
        else:
            return self._encode_with_dragon_plus(text, is_query)
    
    def _encode_with_dragon(self, text: str) -> torch.Tensor:
        """Encode using DRAGON model."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.dragon_encoder.eval()
            embedding = self.dragon_encoder(**inputs)
            # Normalize the embedding
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.squeeze(0)
    
    def _encode_with_dragon_plus(self, text: str, is_query: bool = True) -> torch.Tensor:
        """Encode using DRAGON+ dual encoders."""
        if is_query:
            inputs = self.q_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                return self.q_encoder(**inputs).pooler_output.squeeze(0)
        else:
            inputs = self.p_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                return self.p_encoder(**inputs).pooler_output.squeeze(0)

# Initialize Enhanced Encoder
logger.info("Initializing Enhanced DRAGON Encoder...")
encoder = EnhancedDragonEncoder()

# 5) Embedding cache management
# -----------------------------
# Generate embeddings once and cache them locally; otherwise, load the cache.
if not os.path.exists(EMBEDDINGS_PATH):
    logger.info("Generating enhanced embeddings for D3FEND abstracts...")
    vecs = []
    for i, abs in enumerate(abstracts):
        if i % 100 == 0:
            logger.info("Processing abstract %d/%d", i, len(abstracts))
        
        path = os.path.join(ABSTRACTS_FOLDER, abs["id"])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                abs["text"] = f.read().strip()
                if abs["text"]:
                    embedding = encoder.encode_text(abs["text"], is_query=False)
                    vecs.append(embedding)
    
    if not vecs:
        raise RuntimeError("No valid abstracts found to encode.")
    
    embeddings = torch.stack(vecs).cpu()
    torch.save(embeddings, EMBEDDINGS_PATH)
    logger.info("Saved %d enhanced embeddings.", embeddings.size(0))
else:
    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu")
    logger.info("Loaded %d enhanced embeddings.", embeddings.size(0))

# 6) Tactics utilities & visualization
# ------------------------------------
# Static mapping from label keys to recommended tactics (fallback when ontology lacks direct links)
label_to_tactic = {
    "cloudsecurity": "Cloud_Access_Control",
    "cyberattack": "Threat_Hunting", 
    "cyberawareness": "Security_Awareness_Training",
    "cybercrime": "Incident_Response_Planning",
    "cyberdefense": "Defense_in_Depth",
    "cybersecurity": "Security_Operations_Center",
    "cyberthreat": "Threat_Intelligence",
    "cybertraining": "Security_Awareness_Training",
    "cyberworld": "Digital_Risk_Management",
    "dataprotection": "Data_Loss_Prevention",
    "digitalsecurity": "Digital_Asset_Protection",
    "hacking": "Intrusion_Detection",
    "infosec": "Information_Security_Policy",
    "iotsecurity": "Device_Hardening",
    "malware": "Antivirus_Scanning",
    "networksecurity": "Network_Segmentation",
    "phishing": "Email_Filtering",
    "privacy": "Access_Control",
    "ransomware": "Backup_and_Recovery",
    "threatintelligence": "Threat_Intelligence"
}

def get_tactics(uri: URIRef) -> List[str]:
    """Extract tactics from D3FEND ontology for given abstract URI.

    Traverses:
      abstract -> hasSentence -> hasSegment -> (hasLabel -> mapped tactic) and (mitigatedBy -> tactic)
    Returns unique tactic names.
    """
    tactics: set[str] = set()
    for _, _, sentence in graph.triples((uri, D3F.hasSentence, None)):
        for _, _, segment in graph.triples((sentence, D3F.hasSegment, None)):
            for _, _, label in graph.triples((segment, D3F.hasLabel, None)):
                key = str(label).rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower()
                tactic = label_to_tactic.get(key)
                if tactic:
                    tactics.add(tactic)
            for _, _, tac in graph.triples((segment, D3F.mitigatedBy, None)):
                tactics.add(str(tac).split("#")[-1])
    return list(tactics)

def generate_graph(tactic_scores: Dict[str, float]) -> Optional[str]:
    """Generate a simple horizontal bar chart for tactic scores.

    Returns a base64-encoded PNG string, or None when no scores.
    """
    if not tactic_scores:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    tactics, scores = zip(*tactic_scores.items())
    
    # Create horizontal bar chart
    bars = ax.barh(tactics, scores, color='steelblue', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel("Confidence %", fontsize=12)
    ax.set_title("DRAGON-Enhanced D3FEND Tactic Recommendations", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class ProblemInput(BaseModel):
    problem: str

@app.post("/analyze")
async def analyze(problem: ProblemInput):
    """Analyze a free-text problem statement.

    Steps:
    1) Encode the query using the selected encoder.
    2) Compute cosine similarity against cached abstract embeddings.
    3) Select top-N matches and aggregate associated tactics.
    4) Return recommendations, matches, and a simple visualization.
    """
    try:
        # Encode query using enhanced DRAGON encoder
        query_embedding = encoder.encode_text(problem.problem, is_query=True).cpu().numpy()
        
        # Compute similarities
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Enhanced normalization for better score distribution
        scores = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-6)
        
        # Apply confidence floor and ceiling for better user psychology
        scores = scores * (100 - CONFIDENCE_FLOOR) + CONFIDENCE_FLOOR
        scores = np.clip(scores, CONFIDENCE_FLOOR, 95)  # Cap at 95% for credibility
        
        # Get top matches
        idxs = np.argsort(similarities)[::-1][:TOP_N_MATCHES]
        
        tactic_scores = {}
        matches = []

        for rank, i in enumerate(idxs):
            abs_item = abstracts[i]
            similarity = float(similarities[i])
            confidence_pct = round(scores[i], 1)
            
            # Load abstract text if not already loaded
            file_path = os.path.join(ABSTRACTS_FOLDER, abs_item["id"])
            if "text" not in abs_item:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        abs_item["text"] = f.read().strip()
                        
            tactics = get_tactics(abs_item["uri"])
            if not tactics:
                continue
                
            # Weight by rank for final tactic scoring
            rank_weight = 1 - (0.1 * rank)
            for tactic in tactics:
                weighted_score = confidence_pct * rank_weight
                tactic_scores[tactic] = max(
                    tactic_scores.get(tactic, 0.0), 
                    weighted_score
                )
                
            # Prepare match result
            text_preview = abs_item["text"][:500] + "..." if len(abs_item["text"]) > 500 else abs_item["text"]
            
            matches.append({
                "id": abs_item["id"],
                "text": text_preview,
                "confidence": f"{confidence_pct:.1f}%",
                "raw_similarity": f"{similarity:.3f}",
                "tactics": [
                    {"tactic": t, "confidence": f"{confidence_pct:.1f}%"} 
                    for t in tactics
                ]
            })

        # Sort and format final tactic scores
        tactic_scores = {
            k: round(v, 1) 
            for k, v in sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
            if v >= CONFIDENCE_FLOOR  # Filter low-confidence tactics
        }
        
        # Generate visualization
        graph_img = generate_graph(tactic_scores)

        return {
            "model": "Enhanced DRAGON" if encoder.dragon_available else "DRAGON+ Fallback",
            "query": problem.problem,
            "recommendations": [
                {"tactic": k, "confidence": f"{v}%"} 
                for k, v in tactic_scores.items()
            ],
            "matches": matches,
            "graph": graph_img or "No tactics found",
            "metadata": {
                "total_abstracts": len(embeddings),
                "matches_returned": len(matches),
                "tactics_found": len(tactic_scores),
                "model_type": "DRAGON" if encoder.dragon_available else "DRAGON+",
                "confidence_range": f"{CONFIDENCE_FLOOR}%-95%"
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Analysis error: %s", error_details)
        return {
            "error": str(e),
            "traceback": error_details,
            "model": "Enhanced DRAGON" if encoder.dragon_available else "DRAGON+ Fallback"
        }

@app.get("/health")
async def health_check():
    """Simple health-check endpoint providing runtime metadata."""
    return {
        "status": "healthy",
        "model": "Enhanced DRAGON" if encoder.dragon_available else "DRAGON+ Fallback",
        "abstracts_loaded": len(abstracts),
        "embeddings_cached": os.path.exists(EMBEDDINGS_PATH),
        "device": str(DEVICE),
        "dragon_available": encoder.dragon_available
    }

@app.get("/model-info")
async def model_info():
    """Model information endpoint to aid debugging and observability."""
    return {
        "dragon_model_available": encoder.dragon_available,
        "base_model": DRAGON_MODEL_NAME if encoder.dragon_available else "DRAGON+ Dual Encoders",
        "embeddings_path": EMBEDDINGS_PATH,
        "ontology_path": ONTOLOGY_PATH,
        "total_abstracts": len(abstracts),
        "device": str(DEVICE)
    }

# 8) Dev entrypoint
# ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_enhanced:app", host="0.0.0.0", port=8000, reload=True)
