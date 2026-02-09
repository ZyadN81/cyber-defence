# Cybersecurity Analysis Service

FastAPI backend that analyzes cybersecurity problem descriptions and maps them to MITRE D3FEND tactics using DRAGON/DRAGON+ semantic similarity. The repository is organized into separate backend and frontend folders.

## Repository Structure

```
├── backend/                # FastAPI backend and resources
│   ├── app_enhanced.py     # Main backend service
│   ├── requirements.txt    # Backend dependencies
│   ├── d3fend_output.owl   # D3FEND ontology
│   └── abstracts/          # Local abstract texts (ignored in git)
└── frontend/               # Frontend (if present)
```

## Backend Setup (Windows PowerShell)

- Install dependencies:
  - `pip install -r backend/requirements.txt`
- Run the API:
  - `python backend/app.py`
- API docs:
  - Open http://localhost:8000/docs

## Data Requirements

- Ensure these resources exist:
  - [backend/dragon/](backend/dragon/) — DRAGON/DRAGON+ code and configs (include code; model weights `*.pt` are ignored)
  - [backend/abstracts/](backend/abstracts/) — Local abstract text files (large; kept out of git except `.gitkeep`)
  - [backend/d3fend_output.owl](backend/d3fend_output.owl) — D3FEND ontology file
- Embedding caches (`*.pt`) are generated on first run and are git-ignored

## Frontend

- Place your frontend app in [frontend/](frontend/). Common build outputs are ignored via `.gitignore` (e.g., `node_modules/`, `dist/`, `build/`, `.next/`).
- Run locally (Windows PowerShell):

```powershell
cd frontend
npm install
npm run dev
```

- Build for production:

```powershell
cd frontend
npm run build
```

- Notes:
  - Typical dev servers run on ports like 3000 (Next.js/CRA) or 5173 (Vite). Check your framework output.
  - If your frontend calls the backend API, set the base URL to http://localhost:8000.

## Key Endpoints

- `POST /analyze` — Analyze a problem description and return tactics and matches
- `GET /health` — Health check
- `GET /model-info` — Returns model configuration and availability

## Notes

- First run downloads models and builds embeddings; subsequent runs are faster
- GPU is used if available; otherwise CPU is used
- Large data (abstracts, caches) are ignored via `.gitignore`; use Git LFS if you want to version them
