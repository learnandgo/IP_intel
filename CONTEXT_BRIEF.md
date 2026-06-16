# CONTEXT BRIEF — PatentMind Project
## Transfer document for new Claude conversation/project

---

## WHO I AM (from our conversation)

- Based in **Fremont, CA** (Bay Area)
- Background in **AI, technology consulting, IP analytics**
- Exploring **CCA-F certification** (Claude Certified Architect Foundations)
- Active **stock investor** — semiconductor, space, AI sectors
- Currently applying for an **IP Analyst role** at a company with 1,000+ patent portfolio
- Comfortable with Python, APIs, data science, ML concepts
- Familiar with patent platforms: Innography, AcclaimIP, Derwent, PatSnap, IPRally, Orbit, PatBase
- Studying **Anthropic Academy** (anthropic.skilljar.com) — 19 courses, 3 hrs/day plan

---

## THE PROJECT — PatentMind

**What it is:** A free, local, agentic RAG system for IP Portfolio Management.  
**Purpose:** Interview POC demonstrating all skills in an IP Analyst job description.  
**Dataset:** 10 real Adeia Inc. (NASDAQ: ADEA) patents — already downloaded as PDFs.

### The 10 Adeia Patents (already downloaded)

**Semiconductor Bonding (4):**
- US12300662B2 — DBI to Si Bonding for Simplified Handle Wafer
- US11791307B2 — Microelectronic Component Preparation for Direct Bonding
- US12456709B2 — Structures and Processes for Void-Free Hybrid Bonding
- US11552041B2 — Chemical Mechanical Polishing for Hybrid Bonding

**EPG / Electronic Program Guide (3):**
- US11671648B2 — EPG Information on Display (mini guide + preview bar)
- US9204200B2 — EPG Affinity Clusters (content recommendation)
- US7634792B2 — EPG Signal Acquisition from Multiple Vendors

**Media & Entertainment (3):**
- US11632413B1 — Streaming Media — Adaptive Bitrate
- US12219184B2 — Live Streaming Recommended Content
- US11917214B2 — Live Stream Request and Approval System

### Why Adeia?
- Pure-play IP licensor (spun from Xperi, Oct 2022)
- 11,750+ patents in semiconductor + media/entertainment
- Hybrid bonding patents licensed to Kioxia, Western Digital
- Filed infringement suit vs Disney/Hulu/ESPN (Nov 2024)
- Core technologies: DBI® hybrid bonding, EPG/content discovery, OTT streaming

---

## FILES IN THIS PROJECT

Upload all of these to your new Claude project:

| File | Purpose |
|------|---------|
| `PRD.md` | Full product requirements — features, architecture, demo script, Adeia context |
| `ingest.py` | Patent PDF parser + BAAI embedding + ChromaDB indexer |
| `app.py` | Streamlit dashboard — 5 tabs (Overview, Landscape, Whitespace, Q&A, Reports) |
| `requirements.txt` | All Python dependencies |
| `README.md` | Quick setup guide |
| `CONTEXT_BRIEF.md` | This file — who I am and what we built |

---

## ARCHITECTURE SUMMARY

```
10 Adeia PDFs → ingest.py → ChromaDB (local) → app.py → Streamlit dashboard
                    ↑                                ↑
              pdfplumber                      Claude API
              BAAI/bge-large-en-v1.5         claude-sonnet-4-6
              (free, local embeddings)        (answers + reports)
```

**Total cost: $0** (BAAI model is free/local, ChromaDB is free/local, $5 Anthropic free credit)

---

## SETUP (3 commands)

```bash
pip install -r requirements.txt
python ingest.py --input ./patents/ --format pdf
streamlit run app.py
```

Put the 10 downloaded PDFs in `./patents/` folder first.
Get free Anthropic API key at console.anthropic.com → enter in sidebar.

---

## DASHBOARD TABS

| Tab | What it shows | Job duty demonstrated |
|-----|--------------|----------------------|
| 📊 Portfolio Overview | Filing trends, IPC heatmap, status pie, patent table | Portfolio metrics & visualization |
| 🗺️ Technology Landscape | UMAP cluster map, Claude-labelled clusters | Patent landscaping |
| ⬜ Whitespace Analysis | Coverage gaps vs 20 tech domains, filing recommendations | Whitespace analysis |
| 💬 Patent Q&A | RAG semantic search + Claude answers citing patent IDs | Prior art search, AI workflows |
| 📄 Report Generator | 5 report types, .txt + .docx export | Report automation, M&A diligence |

---

## WHAT TO CONTINUE IN NEW CHAT

Typical next tasks to ask Claude in the new project:

1. **"Run and test the POC"** — Claude Code can run ingest.py and app.py directly
2. **"Fix PDF parsing"** — if pdfplumber misses fields from specific patent PDFs
3. **"Add competitor comparison"** — fetch competitor patents from USPTO/Google Patents API
4. **"Improve the whitespace tab"** — add real IPC code mapping for Adeia's specific domains
5. **"Add citation network"** — visualize forward/backward citations as a graph
6. **"Build the M&A diligence agent"** — auto-fetch a target company's patents and compare
7. **"Connect PatSnap MCP"** — if you get PatSnap free trial credentials
8. **"Generate interview talking points"** — from the PRD demo script

---

## KEY DECISIONS ALREADY MADE

- **ChromaDB** (not Pinecone) — local, free, zero setup
- **BAAI/bge-large-en-v1.5** (not OpenAI embeddings) — free, local, no API cost
- **pdfplumber** (not PyPDF2) — better layout-aware text extraction for patents
- **claude-sonnet-4-6** (not Haiku) — patent analysis needs reasoning depth
- **Embed: title + abstract + claim1 + IPC** (not full text) — avoids signal dilution
- **Streamlit** (not Flask/React) — fastest to build, clean UI, free hosting option

---

## IMPORTANT CONTEXT — OTHER TOPICS FROM OUR CHAT

If relevant to bring up in the new project:

- **Anthropic Academy plan:** 19 courses, 18 days at 3 hrs/day → anthropic.skilljar.com
- **CCA-F exam:** Claude Certified Architect Foundations, $99, scenario-based, 60 questions
- **Patent platforms studied:** Innography, AcclaimIP, Derwent, PatSnap, IPRally, Orbit, PatBase
- **Vector embeddings:** understand how they work (tokenize → neural layers → pooling → cosine similarity)
- **RAG architecture:** query → embed → ChromaDB search → context assembly → Claude → answer
- **MCP:** PatSnap has a live MCP server for Claude Desktop (open.patsnap.com)
- **Bias-variance tradeoff, regression vs classification:** ML fundamentals covered
- **Databricks fraud detection:** built a complete 50K row fraud detection pipeline prompt set

---

## PROMPT TO START NEW CHAT

Paste this at the start of your new Claude project conversation:

> "I'm building PatentMind, a patent portfolio intelligence POC for an IP Analyst job interview. 
> I've uploaded all the project files (PRD.md, app.py, ingest.py, requirements.txt, README.md, CONTEXT_BRIEF.md). 
> I have 10 Adeia patent PDFs already downloaded covering semiconductor bonding, EPG, and media/entertainment. 
> Please read all the uploaded files first, then help me [your next task here]."
