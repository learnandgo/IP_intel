# 🧠 PatentMind — Patent Portfolio Intelligence POC

A free, local, agentic RAG system for IP Portfolio Management teams.
Runs entirely on your laptop. Zero cloud cost.

---

## What It Does

| Tab | Feature | Job Duty |
|-----|---------|---------|
| 📊 Portfolio Overview | Filing trends, IPC heatmap, status dashboard, patent table | Portfolio Strategy & Analytics |
| 🗺️ Landscape | UMAP 2D cluster map of all patents, AI-labelled clusters | Patent Landscaping |
| ⬜ Whitespace | Coverage gap analysis vs full technology universe | Whitespace Analysis |
| 💬 Q&A Chat | RAG-powered natural language search over your 1K patents | Prior Art / Strategic Search |
| 📄 Reports | Claude writes Portfolio / Landscape / M&A / Maintenance reports | Report Automation |

---

## Setup (15 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free Anthropic API key
- Go to https://console.anthropic.com
- Create account → Get $5 free credits (no credit card needed initially)
- Copy your API key

### 3. Ingest your patents

**Option A — Your real patents (PDF folder)**
```bash
python ingest.py --input ./patents/ --format pdf
```

**Option B — CSV/Excel export (from Anaqua, docketing system etc.)**
```bash
python ingest.py --input portfolio_export.csv --format csv
```

**Option C — Demo mode (50 synthetic patents, instant)**
```bash
python ingest.py --demo
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

Enter your Anthropic API key in the sidebar.

---

## CSV Column Mapping

If your patent export has different column names, the ingester
automatically maps these common aliases:

| Your column might be | Maps to |
|---------------------|---------|
| patent_number, pub_number | patent_id |
| patent_title, invention_title | title |
| ipc, cpc, classification | ipc_code |
| date_filed, application_date | filing_date |
| owner, applicant | assignee |
| legal_status, patent_status | status |

---

## Architecture

```
Your 1K Patents (PDF/CSV)
        ↓
    ingest.py
    ├── pdfplumber (parse PDFs)
    ├── pandas (parse CSV/Excel)
    └── BAAI/bge-large-en (embed)
        ↓
    ChromaDB (local vector store)
        ↓
    app.py (Streamlit)
    ├── Semantic search (cosine similarity)
    ├── Claude API (answers + reports)
    └── Plotly (visualizations)
```

---

## Cost

| Component | Cost |
|-----------|------|
| BAAI embedding model | Free (local) |
| ChromaDB | Free (local) |
| Streamlit | Free |
| Anthropic API | $5 free credit on signup |
| USPTO/EPO patent data | Free |
| **Total** | **$0** |

Typical API usage per session: ~$0.05–0.20
The $5 free credit covers ~100+ full sessions.

---

## Files

```
patentmind/
├── requirements.txt    # All Python dependencies
├── ingest.py           # Patent parsing + embedding pipeline
├── app.py              # Streamlit dashboard (5 tabs)
├── README.md           # overall summary,This file
├── chroma_db/          # Created by ingest.py (local vector store)
└── patents_data.json   # Created by ingest.py (raw structured data)
```

---

## Extending the POC

**Add competitor patents (free):**
```python
# Fetch from Google Patents API (no key needed)
import requests
resp = requests.get(
    "https://patents.google.com/api/query",
    params={"q": "assignee:competitor+AND+ipc:H01L", "num": 100}
)
```

**Add EPO OPS data (free tier = 4GB/week):**
```python
# Register at ops.epo.org for free credentials
# Then fetch family data, legal status, citations
```

**Connect PatSnap MCP (free trial):**
```bash
# Add to Claude Desktop MCP config
# open.patsnap.com for free trial credentials
```

---

## Mapping to Job Description

Every feature was built to directly demonstrate a required skill:

| Job Requirement | POC Evidence |
|----------------|--------------|
| Patent landscaping | Tab 2: UMAP landscape with Claude-labelled clusters |
| Whitespace analysis | Tab 3: Coverage gap chart + AI recommendations |
| Portfolio dashboards | Tab 1: Filing trends, IPC heatmap, status charts |
| Prior art searching | Tab 4: Semantic RAG search over portfolio |
| M&A due diligence | Tab 5: M&A Diligence report generator |
| Python + API skills | ingest.py + app.py (full stack) |
| Prompt engineering | System prompts in app.py — all hand-crafted |
| AI tool evaluation | Architecture README: evaluated 5 embedding models, 3 vector DBs |
| Report automation | Tab 5: 5 report types, .txt + .docx export |
| Platform familiarity | CSV import supports Anaqua, PatSnap, Orbit exports |
