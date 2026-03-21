# Cybersecurity Analysis Service

FastAPI backend + Vite frontend for cybersecurity problem analysis mapped to MITRE D3FEND tactics.

## Prerequisites

- Windows PowerShell
- Python 3.10+ (3.12 is supported)
- Node.js 18+ and npm
- Git

## 1) Clone Repository

```powershell
git clone https://github.com/ZyadN81/cyber-defence.git
cd cyber-defence
```

## 2) Run Backend

```powershell
cd backend
python -m pip install -r requirements.txt
python app.py
```

Backend URLs:

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Optional quick health test from another terminal:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

## 3) Run Frontend

Open a new terminal from repo root:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL (usually):

- `http://localhost:5173/`

If port `5173` is busy, Vite automatically uses another port (for example `5174`).

## Required Backend Resources

These must exist in `backend/`:

- `d3fend_output.owl`
- `abstracts/` (abstract text files)
- `dragon/` (DRAGON source files)

The backend validates these at startup and prints explicit setup errors if something is missing.

## First Run Behavior

- Default startup uses a fast local TF-IDF fallback encoder to avoid multi-GB model downloads.
- If `enhanced_dragon_embeddings.pt` does not exist, embeddings/index are generated on first run.
- To force transformer-based startup, set `USE_TRANSFORMERS=1` before running backend.
- Later runs are faster due to local cache reuse.

## API Endpoints

- `POST /analyze`
- `GET /health`
- `GET /model-info`
- `GET /figures`

## Gold Standard Assessment Pack

For thesis assessment-table recomputation and F1 auditability, use:

- `gold_standard_assessment/START_HERE.md`
- `gold_standard_assessment/data/weak_labels_abstracts.csv`
- `gold_standard_assessment/data/manually_validated_gold_subset.csv`
- `gold_standard_assessment/evaluation/f1_recompute_template.xlsx`
- `gold_standard_assessment/evaluation/predictions_template.csv`
- `gold_standard_assessment/reports/gold_standard_summary.json`

Regenerate artifacts from repo root:

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_all.py --sample-size 250 --seed 42
```

Recompute evaluation outputs from manually validated rows:

```powershell
C:/Users/ahmad/AppData/Local/Programs/Python/Python312/python.exe gold_standard_assessment/scripts/run_evaluation.py
```

## Troubleshooting

- `FileNotFoundError` for ontology: run from `backend/` and verify `backend/d3fend_output.owl` exists.
- `No valid abstracts found to encode`: ensure `backend/abstracts/` contains files or provide `backend/enhanced_dragon_embeddings.pt`.
- `vite is not recognized`: run `npm install` inside `frontend/` before `npm run dev`.
