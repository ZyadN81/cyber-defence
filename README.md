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

- First backend run may take time because transformer models can be downloaded.
- If `enhanced_dragon_embeddings.pt` does not exist, embeddings are generated on first run.
- Later runs are faster due to local cache reuse.

## API Endpoints

- `POST /analyze`
- `GET /health`
- `GET /model-info`
- `GET /figures`

## Troubleshooting

- `FileNotFoundError` for ontology: run from `backend/` and verify `backend/d3fend_output.owl` exists.
- `No valid abstracts found to encode`: ensure `backend/abstracts/` contains files or provide `backend/enhanced_dragon_embeddings.pt`.
- `vite is not recognized`: run `npm install` inside `frontend/` before `npm run dev`.
