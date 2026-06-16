# PatentMind — Patent Portfolio Intelligence POC
## Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** June 2026  
**Purpose:** Interview POC for IP Analyst role — demonstrates all required job skills  
**Status:** Ready to build — all code provided

---

## 1. Background & Purpose

This POC was built to demonstrate every skill required in an IP Analyst job description covering:
- IP Portfolio Strategy & Analytics
- Patent Landscaping & Whitespace Analysis
- M&A and IP Due Diligence
- AI Tool Support & Workflow Automation
- Python + API development
- Prompt engineering

The system ingests a real patent portfolio (10 Adeia patents — semiconductor bonding, EPG, media/entertainment) and provides an agentic RAG-powered intelligence dashboard.

---

## 2. Dataset — 10 Adeia Patents

### Semiconductor Bonding (4 patents)
| Patent | Title | Significance |
|--------|-------|-------------|
| US12300662B2 | DBI to Si Bonding for Simplified Handle Wafer | Core hybrid bonding — Kioxia/WD license |
| US11791307B2 | Microelectronic Component Preparation for Direct Bonding | Direct bond interconnect process |
| US12456709B2 | Structures and Processes for Void-Free Hybrid Bonding | Advanced packaging for AI chips |
| US11552041B2 | Chemical Mechanical Polishing for Hybrid Bonding | CMP surface prep for bonding |

### EPG — Electronic Program Guide (3 patents)
| Patent | Title | Significance |
|--------|-------|-------------|
| US11671648B2 | EPG Information on Display — Mini Guide | Core TiVo/Rovi EPG UI patent |
| US9204200B2 | EPG Affinity Clusters | Content recommendation engine |
| US7634792B2 | EPG Signal Acquisition from Multiple Vendors | Multi-source EPG integration |

### Media & Entertainment (3 patents)
| Patent | Title | Significance |
|--------|-------|-------------|
| US11632413B1 | Streaming Media Content — Adaptive Bitrate | OTT streaming core |
| US12219184B2 | Live Streaming Recommended Content | Disney litigation patent |
| US11917214B2 | Live Stream Request and Approval System | Disney litigation patent |

**Download source:** All free from https://patents.google.com/patent/[PATENT_NUMBER]/en

---

## 3. System Architecture

```
YOUR 10 PATENT PDFs
        │
        ▼
   ingest.py
   ├── pdfplumber   → extract text, title, abstract, claims
   ├── pandas       → parse metadata
   └── BAAI/bge-large-en-v1.5  → embed (free, local, 1024-dim)
        │
        ▼
   ChromaDB (local vector store, ./chroma_db/)
   patents_data.json (raw structured data)
        │
        ▼
   app.py (Streamlit, 5 tabs)
   ├── Semantic search (cosine similarity)
   ├── Claude API claude-sonnet-4-6 (answers + reports)
   └── Plotly (visualizations)
```

---

## 4. Feature Specification

### Tab 1 — Portfolio Overview
**Purpose:** Daily KPI dashboard for IP portfolio team  
**Features:**
- KPI cards: Total patents, Granted count, Pending count, Expired count, Tech domains
- Filing trend line chart by year (Plotly)
- Status distribution pie chart
- Top technology domains bar chart (IPC-based)
- Filing activity heatmap (Year × Domain)
- Searchable patent table with filters

**Maps to job duty:** "Track and visualize patent metrics, including filing trends, portfolio statistics"

### Tab 2 — Technology Landscape
**Purpose:** Visual map of portfolio technology clusters  
**Features:**
- UMAP 2D dimensionality reduction of all patent embeddings
- Interactive scatter plot — each dot = one patent
- Color-coded by IPC/technology domain
- Hover shows patent title, filing date, status
- Claude API auto-labels clusters (K-means → cluster names)

**Maps to job duty:** "Perform patent landscaping"

### Tab 3 — Whitespace Analysis
**Purpose:** Identify strategic filing gaps  
**Features:**
- Coverage chart: your portfolio vs 20 technology domains
- Color-coded: Covered (green) / Thin <5 patents (amber) / Gap (red)
- Whitespace gap list
- Claude: "Generate 5 strategic filing recommendations"

**Maps to job duty:** "Whitespace analysis"

### Tab 4 — Patent Q&A Chat
**Purpose:** Natural language search over portfolio  
**Features:**
- RAG pipeline: query → embed → ChromaDB search → top-8 patents → Claude answer
- Cites patent IDs in every response
- Shows retrieved sources with similarity scores
- Example questions pre-loaded
- Chat history with export

**Maps to job duty:** "Prior art searching, AI-assisted IP workflows"

### Tab 5 — Report Generator
**Purpose:** One-click executive-ready reports  
**Report types:**
1. Portfolio Summary — KPIs, tech focus, strategic direction
2. Landscape Report — Technology distribution, IPC analysis
3. Whitespace Analysis — Gap identification, filing opportunities
4. M&A Diligence Summary — Portfolio quality, geographic coverage, risk
5. Maintenance Recommendations — Cost optimization, prune vs retain

**Output formats:** .txt download, .docx download (Word)  
**Maps to job duty:** "Create dashboards and reports, M&A patent diligence, report automation"

---

## 5. Technical Stack (All Free)

| Component | Tool | Version | Cost |
|-----------|------|---------|------|
| Web framework | Streamlit | ≥1.35 | Free |
| Vector database | ChromaDB | ≥0.5 | Free / local |
| Embedding model | BAAI/bge-large-en-v1.5 | via sentence-transformers | Free / local |
| Dimensionality reduction | UMAP | umap-learn ≥0.5.6 | Free |
| LLM / AI | Claude API (claude-sonnet-4-6) | anthropic ≥0.30 | $5 free credit |
| PDF parsing | pdfplumber | ≥0.11 | Free |
| Visualization | Plotly | ≥5.22 | Free |
| Word export | python-docx | ≥1.1 | Free |
| PDF export | reportlab | ≥4.2 | Free |
| Clustering | scikit-learn KMeans | ≥1.5 | Free |
| **Total** | | | **$0** |

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10+
- ~2GB disk space (for BAAI embedding model download, one-time)
- Anthropic API key (free at console.anthropic.com — $5 credit on signup)

### Installation
```bash
# 1. Create project folder
mkdir patentmind && cd patentmind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your patent PDFs
mkdir patents
# Copy your 10 Adeia patent PDFs into ./patents/

# 4. Ingest and embed
python ingest.py --input ./patents/ --format pdf

# 5. Launch dashboard
streamlit run app.py
# Open http://localhost:8501
# Enter Anthropic API key in sidebar
```

### Demo mode (no PDFs needed)
```bash
python ingest.py --demo
streamlit run app.py
```

---

## 7. File Structure

```
patentmind/
├── PRD.md                  ← This document
├── requirements.txt        ← All Python dependencies
├── ingest.py               ← Patent parsing + embedding pipeline
├── app.py                  ← Streamlit dashboard (5 tabs)
├── README.md               ← Quick setup guide
├── patents/                ← Your 10 Adeia PDF files go here
├── chroma_db/              ← Created by ingest.py (local vector store)
└── patents_data.json       ← Created by ingest.py (raw structured data)
```

---

## 8. Key Design Decisions

### Why ChromaDB over Pinecone?
Fully local, zero cost, no cloud account needed. Sufficient for 1K patents.

### Why BAAI/bge-large-en over OpenAI embeddings?
Free, runs locally, no API key, no cost per embedding. 1024 dimensions — excellent quality for technical patent text.

### Why pdfplumber over PyPDF2?
Better text extraction with layout awareness — critical for patent PDFs with multi-column claim sections.

### Why claude-sonnet-4-6 vs Haiku?
Patent analysis requires deep reasoning and nuanced language. Sonnet balances quality and cost. Haiku is too imprecise for claim analysis.

### Embedding strategy
Embed: `Title + Abstract (800 chars) + First Claim (400 chars) + IPC label`  
NOT the full text — full text dilutes signal. Title+abstract+claim1 captures ~90% of retrieval signal.

---

## 9. Interview Demo Script

### Opening (2 min)
"I built PatentMind — a free agentic RAG system for IP portfolio intelligence. It ingests your patent portfolio, embeds every patent using a state-of-the-art local model, and gives your team a natural language interface to your IP."

### Tab 1 Demo (2 min)
"This is the portfolio overview — filing trends, technology distribution, status breakdown. The heatmap shows where filing activity is concentrated. This answers the daily question every IP team has: where is our portfolio strong and where is it thin?"

### Tab 2 Demo (2 min)
"This UMAP landscape map shows every patent as a point. Similar patents cluster together. You can immediately see the three technology clusters — semiconductor bonding here, EPG patents here, streaming media here. Claude auto-labels each cluster."

### Tab 3 Demo (2 min)
"Whitespace analysis. We compare Adeia's portfolio against 20 technology domains. The red bars are gaps — zero coverage. The amber bars are thin. This is how I'd approach a strategic filing recommendation to the R&D team."

### Tab 4 Demo (2 min)
"Natural language Q&A over the portfolio. Watch — I'll ask: 'Which Adeia patents cover hybrid bonding for AI chip packaging?' — it retrieves the 8 most relevant patents and Claude cites specific patent IDs in the answer."

### Tab 5 Demo (1 min)
"One-click report generation. I'll generate the M&A Diligence Summary — Claude writes a structured assessment of Adeia's portfolio that I could hand to a corporate development team today."

### Closing (1 min)
"The entire system is free, runs locally on a laptop, and can be extended to your 1,000+ patent portfolio in about an hour. The same architecture supports PatSnap MCP integration, USPTO bulk data, and EPO OPS for competitive landscaping."

---

## 10. Adeia Business Context (for interview)

**What Adeia is:**
- Pure-play IP licensing company (spun out from Xperi Oct 2022)
- 11,750+ worldwide patent assets
- Two divisions: Semiconductor + Media/Entertainment

**Semiconductor highlights:**
- Pioneer in hybrid bonding / Direct Bond Interconnect (DBI®)
- Licensed to Kioxia, Western Digital (2023), BESI, Micron
- Core technology for AI chip advanced packaging (HBM, chiplets)

**Media/Entertainment highlights:**
- Successor to TiVo, Rovi, Tessera — decades of EPG, content discovery patents
- Filed patent infringement suit vs Disney/Hulu/ESPN (Nov 2024)
- Powers OTT streaming, pay-TV, content recommendation across industry

**Why this portfolio is strategically interesting:**
- Semiconductor bonding patents are increasingly valuable as AI drives chiplet/advanced packaging demand
- EPG patents underpin every streaming platform's content discovery UI
- Active litigation signals these are enforced, high-value assets

---

## 11. Extensions (Post-POC)

| Extension | Effort | Value |
|-----------|--------|-------|
| Add 990 more patents (USPTO bulk download) | 2 hours | Scale to real portfolio |
| PatSnap MCP integration | 1 day | Live competitive data |
| EPO OPS competitor patents | 2 hours | Landscape vs competitors |
| Claim-level analysis agent | 1 day | FTO / validity analysis |
| M&A target comparison | 1 day | Side-by-side diligence |
| Citation network graph | 2 hours | Influence/value mapping |
| Maintenance fee forecaster | 4 hours | Cost optimization |
| Automated weekly digest | 2 hours | Monitoring + alerts |
