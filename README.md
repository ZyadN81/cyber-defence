# Cybersecurity Analysis Service

FastAPI backend that analyzes cybersecurity problem statements and maps them to MITRE D3FEND tactics.

## Quick Start (Windows PowerShell)

### 1) Clone

```powershell
git clone https://github.com/ZyadN81/cyber-defence.git
cd cyber-defence\backend
```

### 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3) Add required backend assets

Before first run, ensure these exist:

- `backend/d3fend_output.owl`
- `backend/abstracts/` with abstract text files
- `backend/dragon/` with DRAGON source files (or run with DRAGON+ fallback if DRAGON code is unavailable)

If `backend/abstracts/` is empty, provide a prebuilt embedding cache at:

- `backend/enhanced_dragon_embeddings.pt`

### 4) Run backend

```powershell
python app.py
```

API docs: http://localhost:8000/docs

## Frontend Run

```powershell
cd ..\frontend
npm install
npm run dev
```

## API Endpoints

- `POST /analyze`
- `GET /health`
- `GET /model-info`
- `GET /figures`

## Notes

- Run backend from `backend/` using `python app.py`.
- First run can take longer because models/downloads and embedding generation happen at startup.
- If startup fails, the backend now prints explicit missing-file setup messages.
