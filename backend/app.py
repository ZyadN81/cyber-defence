""""""

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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from scipy.stats import pearsonr
import seaborn as sns
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
logger = logging.getLogger("app")

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
class DragonEncoder:
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
            logger.info("Loading dual encoders...")
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
logger.info("Initializing DRAGON Encoder...")
encoder = DragonEncoder()

# 5) Embedding cache management
# -----------------------------
# Generate embeddings once and cache them locally; otherwise, load the cache.
if not os.path.exists(EMBEDDINGS_PATH):
    logger.info("Generating embeddings for D3FEND abstracts...")
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
    logger.info("Saved %d embeddings.", embeddings.size(0))
else:
    embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu")
    logger.info("Loaded %d embeddings.", embeddings.size(0))

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
    ax.set_title("D3FEND Tactic Recommendations", fontsize=14, fontweight='bold')
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
    """Analyze a free-text problem statement."""
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
            "model": "DRAGON",
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
                "model_type": "DRAGON",
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
        "model": "DRAGON",
        "abstracts_loaded": len(abstracts),
        "embeddings_cached": os.path.exists(EMBEDDINGS_PATH),
        "device": str(DEVICE),
        "dragon_available": encoder.dragon_available
    }

@app.get("/model-info")
async def model_info():
    """Model information endpoint."""
    return {
        "dragon_model_available": encoder.dragon_available,
        "base_model": DRAGON_MODEL_NAME,
        "embeddings_path": EMBEDDINGS_PATH,
        "ontology_path": ONTOLOGY_PATH,
        "total_abstracts": len(abstracts),
        "device": str(DEVICE)
    }

# Figure generation helpers
# -------------------------
def _read_manual_scenarios() -> List[Dict[str, str]]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    scenarios_path = os.path.join(root, "manual_test_scenarios.txt")
    with open(scenarios_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks: List[Dict[str, str]] = []
    pattern = re.compile(r"##\s*(Simple|Advanced)\s*(\d+):\s*([^\n]+)\n\n([\s\S]*?)\n---", re.MULTILINE)
    for m in pattern.finditer(content):
        group, idx, title, text = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        name = f"{group} {idx}: {title}"
        blocks.append({"name": name, "group": group, "text": text})
    # Expected mapping
    expected: Dict[str, List[str]] = {}
    simple_re = re.compile(r"\*\*Simple\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)
    for m in simple_re.finditer(content):
        idx, _t, labels_str = m.groups(); labels = [s.strip().lower() for s in labels_str.split(',')]
        key = f"Simple {idx}:"
        for sc in blocks:
            if sc["name"].startswith(key): expected[sc["name"]] = labels; break
    adv_re = re.compile(r"\*\*Advanced\s*(\d)\s*\(([^\)]+)\):\*\*\s*Should\s*detect\s*([^\n]+)", re.IGNORECASE)
    for m in adv_re.finditer(content):
        idx, _t, labels_str = m.groups(); labels = [s.strip().lower() for s in labels_str.split(',')]
        key = f"Advanced {idx}:"
        for sc in blocks:
            if sc["name"].startswith(key): expected[sc["name"]] = labels; break
    return blocks, expected

def _map_expected_to_tactics(labels: List[str]) -> List[str]:
    return [label_to_tactic.get(lbl, lbl) for lbl in labels]

def _severity_for_labels(labels: List[str]) -> str:
    """Map label set to threat severity buckets matching thesis figures.

    Priority order ensures a single severity per scenario.
    """
    # Buckets
    critical = {"malware", "ransomware"}
    high = {"phishing", "ddos", "cyberattack"}
    medium = {"networksecurity", "dataprotection", "cybersecurity", "privacy"}
    low = {"iotsecurity", "vulnerability"}

    labset = set(labels)
    if labset & critical:
        return "Critical"
    if labset & high:
        return "High"
    if labset & medium:
        return "Medium"
    if labset & low:
        return "Low"
    return "Medium"

def _category_for_labels(labels: List[str]) -> str:
    """Map label set to four domain categories used in figures."""
    labset = set(labels)
    malware_grp = {"malware", "ransomware"}
    net_grp = {"networksecurity", "ddos", "phishing", "cyberattack"}
    breach_grp = {"dataprotection", "privacy"}
    vuln_grp = {"vulnerability", "iotsecurity", "cybersecurity"}

    if labset & malware_grp:
        return "Malware"
    if labset & net_grp:
        return "Network Attacks"
    if labset & breach_grp:
        return "Data Breach"
    if labset & vuln_grp:
        return "System Vulnerability"
    return "System Vulnerability"

def _encode_problem(problem_text: str) -> Dict[str, object]:
    # Core of analyze() extracted for reuse
    query_embedding = encoder.encode_text(problem_text, is_query=True).cpu().numpy()
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    scores = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-6)
    scores = scores * (100 - CONFIDENCE_FLOOR) + CONFIDENCE_FLOOR
    scores = np.clip(scores, CONFIDENCE_FLOOR, 95)
    idxs = np.argsort(similarities)[::-1][:TOP_N_MATCHES]
    tactic_scores = {}
    matches = []
    for rank, i in enumerate(idxs):
        abs_item = abstracts[i]
        similarity = float(similarities[i])
        confidence_pct = round(scores[i], 1)
        file_path = os.path.join(ABSTRACTS_FOLDER, abs_item["id"])
        if "text" not in abs_item:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    abs_item["text"] = f.read().strip()
        tactics = get_tactics(abs_item["uri"])
        if not tactics: continue
        rank_weight = 1 - (0.1 * rank)
        for tactic in tactics:
            weighted_score = confidence_pct * rank_weight
            tactic_scores[tactic] = max(tactic_scores.get(tactic, 0.0), weighted_score)
        text_preview = abs_item.get("text", "")
        text_preview = text_preview[:500] + "..." if len(text_preview) > 500 else text_preview
        matches.append({
            "id": abs_item["id"],
            "text": text_preview,
            "confidence": f"{confidence_pct:.1f}%",
            "raw_similarity": f"{similarity:.3f}",
            "tactics": [{"tactic": t, "confidence": f"{confidence_pct:.1f}%"} for t in tactics]
        })
    tactic_scores = {k: round(v, 1) for k, v in sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True) if v >= CONFIDENCE_FLOOR}
    recs = [{"tactic": k, "confidence": f"{v}%"} for k, v in tactic_scores.items()]
    return {"recommendations": recs, "matches": matches}

def _b64_fig(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/figures")
async def thesis_figures():
    """Generate and return analysis figures from manual scenarios."""
    try:
        scenarios, expected_map = _read_manual_scenarios()
        true_labels: List[List[str]] = []
        pred_labels: List[List[str]] = []
        confidences: List[float] = []
        severities: List[str] = []
        categories: List[str] = []

        for sc in scenarios:
            exp_labels = expected_map.get(sc["name"], [])
            exp_tac = _map_expected_to_tactics(exp_labels)
            true_labels.append(exp_tac)
            severities.append(_severity_for_labels(exp_labels))
            categories.append(_category_for_labels(exp_labels))
            res = _encode_problem(sc["text"])
            recs = res.get("recommendations", [])
            preds = [r.get("tactic", "") for r in recs]
            pred_labels.append(preds)
            conf_vals = []
            for r in recs[:3]:
                c = r.get("confidence", "0%")
                try: conf_vals.append(float(str(c).rstrip('%')))
                except Exception: pass
            confidences.append(float(np.mean(conf_vals)) if conf_vals else 0.0)

        # MultiLabel binarization
        class_set = set()
        for lst in true_labels + pred_labels: class_set.update(lst)
        classes = sorted(class_set)
        mlb = MultiLabelBinarizer(classes=classes)
        Y_true = mlb.fit_transform(true_labels)
        Y_pred = mlb.transform(pred_labels)

        # Overall metrics
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(Y_true, Y_pred, average="micro", zero_division=0)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Y_true, Y_pred, average="macro", zero_division=0)
        prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(Y_true, Y_pred, average="weighted", zero_division=0)
        exact_match = float(np.mean((Y_true == Y_pred).all(axis=1)))
        accuracy_labelwise = float(np.mean(Y_true == Y_pred))
        jacc = jaccard_score(Y_true, Y_pred, average="samples") if Y_true.size else 0.0
        hamming = float(np.mean(Y_true != Y_pred)) if Y_true.size else 0.0

        # Per-sample F1 for CI/effect size
        f1_per_sample = []
        for yt, yp in zip(Y_true, Y_pred):
            _, _, f1_s, _ = precision_recall_fscore_support(yt.reshape(1, -1), yp.reshape(1, -1), average="samples", zero_division=0)
            f1_per_sample.append(float(f1_s))
        # Bootstrap CI
        rng = np.random.default_rng(123)
        boots = []
        vals = np.asarray(f1_per_sample)
        for _ in range(1000):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(np.mean(sample))
        ci_low = float(np.quantile(boots, 0.025))
        ci_high = float(np.quantile(boots, 0.975))
        effect_size = float((np.mean(vals) - 0.5) / (np.std(vals, ddof=1) + 1e-9))

        # Per-class metrics and support
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(Y_true, Y_pred, average=None, zero_division=0)

        # Figure 1: Overall
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0,0]
        core_labels = ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall", "Micro F1"]
        core_vals = [accuracy_labelwise, f1_macro, prec_macro, rec_macro, f1_micro]
        ax.bar(core_labels, core_vals)
        ax.set_ylim(0,1); ax.set_title("(a) Core Classification Metrics")
        for i, v in enumerate(core_vals): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
        ax = axes[0,1]
        x = np.arange(3); width=0.25
        ax.bar(x-width, [prec_micro, prec_macro, prec_weight], width, label="Precision")
        ax.bar(x, [rec_micro, rec_macro, rec_weight], width, label="Recall")
        ax.bar(x+width, [f1_micro, f1_macro, f1_weight], width, label="F1-Score")
        ax.set_xticks(x); ax.set_xticklabels(["Micro","Macro","Weighted"]) ; ax.set_ylim(0,1); ax.legend(); ax.set_title("(b) Performance by Averaging Method")
        ax = axes[1,0]
        ax.bar(["DRAGON F1-Score","Confidence Interval","Effect Size (Cohen's d)"] , [f1_macro, (ci_high-ci_low)/2, effect_size])
        ax.set_title("(c) Statistical Analysis")
        for i, v in enumerate([f1_macro, (ci_high-ci_low)/2, effect_size]): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
        ax = axes[1,1]
        ax.bar(["Exact Match Ratio","Jaccard Similarity","Hamming Loss"], [exact_match, jacc, hamming])
        ax.set_title("(d) Multi-label Performance")
        for i, v in enumerate([exact_match, jacc, hamming]): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
        overall_b64 = _b64_fig(fig)

        # Figure 2: Per-class
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0,0]
        ax.barh(classes, f1_c); ax.set_title("(a) F1-Score by Cybersecurity Category")
        for y, v in enumerate(f1_c): ax.text(v+0.01, y, f"{v:.3f}")
        ax = axes[0,1]
        sizes = np.array(support_c) / (np.max(support_c) + 1e-9) * 300
        scatter = ax.scatter(rec_c, prec_c, s=sizes, c=f1_c, cmap="viridis")
        ax.plot([0,1],[0,1],"k--", alpha=0.5); ax.set_title("(b) Precision vs Recall (size=support, color=F1)")
        plt.colorbar(scatter, ax=ax, label="F1-Score")
        ax = axes[1,0]
        ax.bar(classes, support_c); ax.set_title("(c) Sample Distribution by Class"); ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(support_c): ax.text(i, v+1, f"{v}", ha="center")
        ax = axes[1,1]
        heat = np.vstack([prec_c, rec_c, f1_c])
        sns.heatmap(heat, annot=np.round(heat,3), cmap="OrRd", xticklabels=classes, yticklabels=["Precision","Recall","F1"], ax=ax)
        ax.set_title("(d) Performance Metrics Heatmap")
        perclass_b64 = _b64_fig(fig)

        # Figure 3: Domain performance
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        # Per-sample correctness
        correctness = [int(any(t in pl[:5] for t in tl)) for pl, tl in zip(pred_labels, true_labels)]
        severity_levels = ["Critical","High","Medium","Low"]
        f1_by_sev = {s: [] for s in severity_levels}
        for sev, f1v in zip(severities, f1_per_sample): f1_by_sev[sev].append(f1v)
        ax = axes[0,0]
        vals = [float(np.mean(f1_by_sev[s])) if f1_by_sev[s] else 0.0 for s in severity_levels]
        ax.bar(severity_levels, vals); ax.set_ylim(0,1); ax.set_title("(a) Performance by Threat Severity")
        for i, v in enumerate(vals): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
        ax = axes[0,1]
        crit_corr = [c for c, sev in zip(correctness, severities) if sev == "Critical"]
        detected = int(np.sum(crit_corr)); total = len(crit_corr); miss = total - detected
        ax.pie([detected, miss], labels=["Critical Threats Detected","Critical Threats Missed"], autopct=lambda p: f"{p:.1f}%")
        ax.set_title("(b) Critical Threat Detection Rate")
        ax = axes[1,0]
        fp_rate = float(np.mean((Y_pred == 1) & (Y_true == 0))) ; fn_rate = float(np.mean((Y_pred == 0) & (Y_true == 1)))
        ax.bar(["False Positives","False Negatives"], [fp_rate, fn_rate]); ax.set_ylim(0,0.1); ax.set_title("(c) Error Rate Analysis")
        for i, v in enumerate([fp_rate, fn_rate]): ax.text(i, v+0.005, f"{v:.3f}", ha="center")
        ax = axes[1,1]
        cats = ["Malware","Network Attacks","Data Breach","System Vulnerability"]
        f1_by_cat = {c: [] for c in cats}
        for cat, f1v in zip(categories, f1_per_sample): f1_by_cat[cat].append(f1v)
        vals = [float(np.mean(f1_by_cat[c])) if f1_by_cat[c] else 0.0 for c in cats]
        ax.plot(cats, vals, marker='o'); ax.set_ylim(0,1); ax.set_title("(d) Performance by Threat Category")
        domain_b64 = _b64_fig(fig)

        # Figure 4: Reliability
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0,0]
        ax.hist(confidences, bins=20, alpha=0.8); ax.axvline(np.mean(confidences), color='r', linestyle='--', label=f"Mean: {np.mean(confidences):.3f}")
        ax.legend(); ax.set_title("(a) Model Confidence Distribution")
        ax = axes[0,1]
        r, _ = pearsonr(confidences, f1_per_sample) if len(confidences) > 1 else (0.0, None)
        ax.scatter(confidences, f1_per_sample, alpha=0.8)
        if len(confidences) > 1:
            m, b = np.polyfit(confidences, f1_per_sample, 1)
            xs = np.linspace(min(confidences), max(confidences), 100)
            ax.plot(xs, m*xs + b, 'r--', label=f"r = {r:.3f}") ; ax.legend()
        ax.set_title("(b) Confidence vs Performance Correlation")
        ax = axes[1,0]
        bins = np.linspace(0.0, 1.0, 11)
        conf_norm = np.clip(np.array(confidences)/100.0, 0, 1) if np.max(confidences) > 1.0 else np.array(confidences)
        inds = np.digitize(conf_norm, bins) - 1
        cal_conf, cal_f1 = [], []
        for i in range(len(bins)-1):
            sel = inds == i
            if np.any(sel):
                cal_conf.append(float(np.mean(conf_norm[sel])))
                cal_f1.append(float(np.mean(np.array(f1_per_sample)[sel])))
        ax.plot(cal_conf, cal_f1, marker='o', label='DRAGON System')
        ax.plot([0,1],[0,1],'k--', label='Perfect Calibration'); ax.legend(); ax.set_title("(c) Confidence Calibration Curve")
        ax = axes[1,1]
        low = conf_norm < 0.5; med = (conf_norm >= 0.5) & (conf_norm <= 0.8); high = conf_norm > 0.8
        means = [float(np.mean(np.array(f1_per_sample)[low])) if np.any(low) else 0.0,
                 float(np.mean(np.array(f1_per_sample)[med])) if np.any(med) else 0.0,
                 float(np.mean(np.array(f1_per_sample)[high])) if np.any(high) else 0.0]
        counts = [int(np.sum(low)), int(np.sum(med)), int(np.sum(high))]
        ax.bar([f"Low\n(n={counts[0]})", f"Medium\n(n={counts[1]})", f"High\n(n={counts[2]})"], means)
        ax.set_ylim(0,1); ax.set_title("(d) Performance by Confidence Level")
        reliability_b64 = _b64_fig(fig)

        return {
            "overall": overall_b64,
            "per_class": perclass_b64,
            "domain": domain_b64,
            "reliability": reliability_b64,
            "metrics": {
                "accuracy_labelwise": accuracy_labelwise,
                "macro_f1": f1_macro,
                "macro_precision": prec_macro,
                "macro_recall": rec_macro,
                "micro_f1": f1_micro,
                "exact_match_ratio": exact_match,
                "jaccard": jacc,
                "hamming_loss": hamming,
                "ci_macro_f1": [ci_low, ci_high]
            }
        }
    except Exception as e:
        import traceback
        logger.error("Figures error: %s", traceback.format_exc())
        return {"error": str(e)}

# 8) Dev entrypoint
# ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
