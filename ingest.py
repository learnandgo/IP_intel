"""
PatentMind — Patent Portfolio Intelligence POC
============================================================
Step 1: Ingest your 1,000 patents and build the vector index.

Supports:
  - PDF files  (pdfplumber extracts text + metadata)
  - CSV/Excel  (bulk export from Anaqua, Docketing systems etc.)
  - JSON       (PatSnap, Orbit, or custom exports)

Usage:
  python ingest.py --input ./patents/ --format pdf
  python ingest.py --input patents.csv --format csv
"""

import os, json, re, argparse, hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "patents"
EMBED_MODEL   = "Snowflake/snowflake-arctic-embed-l-v2.0"   # evaluated against BAAI/bge-large-en-v1.5 free local model — 1024 dims
BATCH_SIZE    = 32                           # embed 32 patents at a time

# ── IPC → plain-English lookup (top 50 codes relevant to media/semiconductor) ─
IPC_LABELS = {
    "H01L": "Semiconductor Devices",
    "H04N": "Pictorial Communication / TV",
    "H04L": "Digital Data Transmission",
    "H04W": "Wireless Communication",
    "G06F": "Electric Digital Data Processing",
    "G06N": "Computing Models / AI/ML",
    "G06T": "Image Processing",
    "G11C": "Static Data Storage",
    "H03M": "Coding / Decoding",
    "H04B": "Radio Transmission",
    "H04J": "Multiplex Communication",
    "H04K": "Secret Communication",
    "H04S": "Stereophonic Systems",
    "H04R": "Loudspeakers / Microphones",
    "G10L": "Speech / Audio Processing",
    "G11B": "Information Storage (magnetic/optical)",
    "H01S": "Lasers",
    "H02M": "Power Conversion",
    "G06K": "Recognition of Data",
    "G06V": "Image / Video Recognition",
}

def get_ipc_label(code: str) -> str:
    if not code:
        return "Unknown"
    prefix = code[:4].upper()
    return IPC_LABELS.get(prefix, prefix)

# ── PARSERS ───────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> dict:
    """Extract structured data from a USPTO patent PDF using pymupdf."""
    try:
        import fitz
        doc = fitz.open(str(path))
        pages_text = []
        for page in doc[:12]:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        full_text = "\n".join(pages_text)
        doc.close()
    except Exception as e:
        print(f"    ⚠️  pymupdf error: {e}")
        full_text = ""

    def find(pattern, text, group=1, default=""):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(group).strip()[:500] if m else default

    def clean(s):
        """Fix USPTO OCR spacing: 'C H E M I C A L' → 'CHEMICAL'"""
        s = re.sub(r'(?<=[A-Z])\s(?=[A-Z\d])', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    # ── TITLE: USPTO field (54) ───────────────────────────────────────────────
    title = (
        find(r'\(\s*54\s*\)\s+([A-Z][^\n()]{10,300})', full_text)
        or find(r'\(\s*54\s*\)\s*\n+\s*([^\n()]{10,300})', full_text)
        or find(r'(?:title of invention|invention title)[:\s]+([^\n]{10,200})', full_text)
        or path.stem
    )
    title = clean(title)

    # ── ABSTRACT: USPTO field (57) ────────────────────────────────────────────
    abstract = (
        find(r'\(\s*57\s*\)\s*(?:ABSTRACT|Abstract)?\s*\n+([\s\S]{50,2000}?)(?:\n{3,}|\(\d+\)|^\s*1\.)', full_text)
        or find(r'(?:ABSTRACT|Abstract)\s*\n+([\s\S]{50,2000}?)(?:\n{3,}|(?:CLAIMS?|What is claimed))', full_text)
        or ""
    )

    # ── CLAIMS ────────────────────────────────────────────────────────────────
    claims = (
        find(r'(?:CLAIMS?|What is claimed\s*is?)\s*\n+([\s\S]{50,3000}?)(?:\n{4,}|ABSTRACT|DESCRIPTION)', full_text)
        or find(r'\n\s*1\.\s+((?:A|An|The)\s+(?:method|system|apparatus|device|process).{20,})', full_text)
        or ""
    )

    # ── IPC CODE: USPTO field (51) ────────────────────────────────────────────
    ipc = (
        find(r'\(\s*51\s*\).*?([A-H]\d{2}[A-Z]\s*\d+/\d+)', full_text)
        or find(r'(?:Int\.?\s*Cl\.?)[:\s]+([A-H]\d{2}[A-Z]\s*\d+/\d+)', full_text)
        or ""
    )
    # Fix common OCR errors: letter O → zero, remove spaces
    ipc = re.sub(r'\s+', '', ipc)
    ipc = re.sub(r'(?<=[A-Z])O(?=\d)', '0', ipc)

    # ── PATENT NUMBER ─────────────────────────────────────────────────────────
    pat_num = (
        find(r'Patent\s+No\.\s*:\s*\n?\s*US\s*([\d,]+\s*[A-Z]\d?)', full_text)
        or find(r'US\s*0*(\d{7,8}\s*[A-Z]\d?)', full_text)
        or path.stem
    )
    pat_num = pat_num.replace(",", "").replace(" ", "")
    if not pat_num.startswith("US"):
        pat_num = "US" + pat_num

    # ── FILING DATE: USPTO field (22) ─────────────────────────────────────────
    filing = (
        find(r'\(\s*22\s*\)\s*(?:Filed\s*:?)?\s*([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})', full_text)
        or find(r'(?:Filed|Filing Date)\s*[:\s]+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})', full_text)
        or find(r'\(\s*22\s*\).*?(\d{4}-\d{2}-\d{2})', full_text)
        or ""
    )

    # ── ASSIGNEE: USPTO field (73) ────────────────────────────────────────────
    assignee = (
        find(r'\(\s*73\s*\)\s*Assignee\s*:\s*([^\n()]{5,150})', full_text)
        or find(r'(?:Assignee|Assigned to)\s*[:\s]+([^\n]{5,100})', full_text)
        or "Adeia Inc."
    )
    assignee = clean(assignee)

    print(f"    Title:    {title[:70]}")
    print(f"    Abstract: {abstract[:80] if abstract else 'EMPTY ⚠️'}")
    print(f"    IPC:      {ipc or 'not found ⚠️'}")
    print(f"    Filed:    {filing or 'not found'}")

    return {
        "patent_id":   pat_num,
        "title":       title,
        "abstract":    abstract or full_text[200:900],
        "claims":      claims,
        "ipc_code":    ipc,
        "ipc_label":   get_ipc_label(ipc),
        "filing_date": filing,
        "assignee":    assignee,
        "status":      "granted",
        "source_file": str(path),
        "full_text":   full_text[:4000],
    }


def parse_csv(path: Path) -> list[dict]:
    """
    Parse CSV / Excel patent export.
    Expected columns (flexible — we map common aliases):
      patent_id | title | abstract | ipc_code | filing_date | assignee | status
    """
    df = pd.read_excel(path) if path.suffix in [".xlsx", ".xls"] else pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    col_map = {
        "patent_number": "patent_id", "pub_number": "patent_id",
        "publication_number": "patent_id", "app_number": "patent_id",
        "patent_title": "title", "invention_title": "title",
        "ipc": "ipc_code", "ipc_classification": "ipc_code",
        "cpc": "ipc_code", "classification": "ipc_code",
        "date_filed": "filing_date", "application_date": "filing_date",
        "grant_date": "filing_date",
        "owner": "assignee", "applicant": "assignee",
        "legal_status": "status", "patent_status": "status",
    }
    df = df.rename(columns=col_map)

    for col in ["patent_id", "title", "abstract", "ipc_code",
                "filing_date", "assignee", "status"]:
        if col not in df.columns:
            df[col] = ""

    records = []
    for _, row in df.iterrows():
        ipc = str(row.get("ipc_code", ""))
        records.append({
            "patent_id":   str(row.get("patent_id", f"PAT_{_}")),
            "title":       str(row.get("title", "")),
            "abstract":    str(row.get("abstract", "")),
            "claims":      str(row.get("claims", "")),
            "ipc_code":    ipc,
            "ipc_label":   get_ipc_label(ipc),
            "filing_date": str(row.get("filing_date", "")),
            "assignee":    str(row.get("assignee", "Your Company")),
            "status":      str(row.get("status", "unknown")),
            "source_file": str(path),
            "full_text":   f"{row.get('title','')} {row.get('abstract','')} {row.get('claims','')}",
        })
    return records


def load_patents(input_path: str, fmt: str) -> list[dict]:
    """Load all patents from the given path."""
    p = Path(input_path)
    patents = []

    if fmt == "csv" or (p.is_file() and p.suffix in [".csv", ".xlsx", ".xls"]):
        patents = parse_csv(p)

    elif fmt == "pdf" or p.is_dir():
        files = list(p.glob("**/*.pdf")) if p.is_dir() else [p]
        print(f"Found {len(files)} PDF files")
        for i, f in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] Parsing {f.name}...")
            patents.append(parse_pdf(f))

    elif fmt == "json" or (p.is_file() and p.suffix == ".json"):
        with open(p) as f:
            data = json.load(f)
        patents = data if isinstance(data, list) else [data]

    print(f"\n✅ Loaded {len(patents)} patents")
    return patents


# ── EMBEDDING & INDEXING ──────────────────────────────────────────────────────

def build_embed_text(patent: dict) -> str:
    """
    What we embed: title + abstract + first claim.
    This is the key retrieval text — determines search quality.
    """
    parts = [
        f"Title: {patent.get('title', '')}",
        f"Abstract: {patent.get('abstract', '')[:800]}",
        f"Claims: {patent.get('claims', '')[:400]}",
        f"IPC: {patent.get('ipc_label', '')}",
    ]
    return " | ".join(p for p in parts if len(p) > 10)


def ingest(patents: list[dict]):
    """Embed all patents and store in ChromaDB."""
    print(f"\n⚙️  Loading embedding model: {EMBED_MODEL}")
    print("   (First run downloads ~1.3GB — subsequent runs are instant)\n")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    texts, ids, metadatas = [], [], []
    for p in patents:
        pid = hashlib.md5(str(p.get("patent_id", "") + p.get("title", "")).encode()).hexdigest()[:16]
        texts.append(build_embed_text(p))
        ids.append(pid)
        metadatas.append({
            "patent_id":   str(p.get("patent_id", ""))[:100],
            "title":       str(p.get("title", ""))[:200],
            "abstract":    str(p.get("abstract", ""))[:500],
            "ipc_code":    str(p.get("ipc_code", ""))[:50],
            "ipc_label":   str(p.get("ipc_label", ""))[:100],
            "filing_date": str(p.get("filing_date", ""))[:50],
            "assignee":    str(p.get("assignee", ""))[:200],
            "status":      str(p.get("status", "unknown"))[:50],
        })

    # Embed in batches
    print(f"🔢 Embedding {len(patents)} patents in batches of {BATCH_SIZE}...")
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        print(f"   Embedded {min(i+BATCH_SIZE, len(patents))}/{len(patents)}")

    # Upsert to ChromaDB
    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"\n✅ Indexed {len(patents)} patents → {CHROMA_PATH}/")
    print(f"   Run: streamlit run app.py")

    # Save raw data as JSON for dashboard use
    with open("patents_data.json", "w") as f:
        json.dump(patents, f, indent=2, default=str)
    print(f"   Saved raw data → patents_data.json")


# ── DEMO DATA GENERATOR ───────────────────────────────────────────────────────

def generate_demo_patents(n: int = 50) -> list[dict]:
    """Generate synthetic patents for demo if you don't have real data yet."""
    import random
    random.seed(42)

    domains = [
        ("H04N", "Video Coding", "A method for encoding video streams using adaptive bitrate algorithms"),
        ("H01L", "FinFET Transistor", "A semiconductor device comprising fin-based field effect transistors"),
        ("H04W", "5G Beamforming", "A system for directional signal transmission in millimeter wave networks"),
        ("G06F", "Cache Coherence", "A multi-processor cache coherence protocol for distributed memory"),
        ("H03M", "LDPC Coding", "Low-density parity-check encoder/decoder for error correction"),
        ("G06T", "Image Compression", "Neural network based image compression with perceptual quality metrics"),
        ("G06N", "Neural Architecture", "Transformer-based architecture for efficient edge inference"),
        ("G11C", "DRAM Cell", "Dynamic random access memory cell with reduced leakage current"),
        ("H04L", "Packet Routing", "Quality of service packet routing for multimedia streaming"),
        ("G10L", "Speech Codec", "Wideband speech codec with voice activity detection"),
    ]

    patents = []
    for i in range(n):
        ipc, topic, abstract_base = random.choice(domains)
        year = random.randint(2015, 2024)
        month = random.randint(1, 12)
        patents.append({
            "patent_id":   f"US{10000000 + i:08d}",
            "title":       f"{topic} System and Method — Variant {i+1}",
            "abstract":    f"{abstract_base}. The invention provides improvements in {random.choice(['efficiency', 'throughput', 'latency', 'power consumption', 'accuracy'])} by {random.choice(['optimizing', 'reducing', 'enhancing', 'implementing'])} the core {topic.lower()} pipeline.",
            "claims":      f"1. A method comprising: receiving input data; processing using {topic.lower()} algorithm; outputting result with improved {random.choice(['quality', 'speed', 'efficiency'])}.",
            "ipc_code":    ipc,
            "ipc_label":   get_ipc_label(ipc),
            "filing_date": f"{year}-{month:02d}-{random.randint(1,28):02d}",
            "assignee":    "Your Company Inc.",
            "status":      random.choice(["granted", "granted", "granted", "pending", "expired"]),
            "source_file": f"demo_patent_{i}.pdf",
            "full_text":   f"{topic} {abstract_base}",
        })
    return patents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatentMind Ingestion Pipeline")
    parser.add_argument("--input",  default="./patents/", help="Path to patents folder or CSV file")
    parser.add_argument("--format", default="auto", choices=["pdf", "csv", "json", "demo"])
    parser.add_argument("--demo",   action="store_true", help="Use 50 synthetic demo patents")
    args = parser.parse_args()

    if args.demo or args.format == "demo":
        print("📋 Generating 50 demo patents for POC...")
        patents = generate_demo_patents(50)
    else:
        patents = load_patents(args.input, args.format)

    ingest(patents)
