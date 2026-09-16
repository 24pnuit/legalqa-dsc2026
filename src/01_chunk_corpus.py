import json
import re
import glob
import unicodedata
import os

CONTEXTS_DIR = "data/contexts/selected-contexts"
OUT_PATH = "data/chunks.jsonl"

# --- Cleaning ---
def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    # collapse repeated whitespace/newlines but keep paragraph breaks somewhat
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r\n|\r", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()

# Matches "Điều 12." / "Điều 12a." / "Điều 12." at start of a line
ARTICLE_RE = re.compile(r"(?m)^(Điều\s+\d+[a-zA-Z]?\.)")

MAX_CHUNK_CHARS = 3000  # ~ safe bound before we sub-split an oversized article

def split_into_articles(doc_text: str):
    """Split a full legal document into (article_label, article_text) pieces.
    Falls back to fixed-size sliding windows if no 'Điều' markers are found."""
    matches = list(ARTICLE_RE.finditer(doc_text))
    if not matches:
        return None  # signal caller to use fallback windowing

    pieces = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        article_label = m.group(1).rstrip(".")
        article_text = doc_text[start:end].strip()
        pieces.append((article_label, article_text))
    return pieces


def sliding_window_fallback(doc_text: str, window_chars=1800, overlap_chars=200):
    pieces = []
    i = 0
    n = len(doc_text)
    idx = 0
    while i < n:
        chunk = doc_text[i:i + window_chars]
        pieces.append((f"phần_{idx}", chunk))
        idx += 1
        i += window_chars - overlap_chars
    return pieces


def maybe_subsplit_article(label: str, text: str):
    """If a single Điều is unusually long, sub-split by Khoản (numbered clauses like '1.' '2.')."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [(label, text)]
    # try splitting on lines that look like "1. ", "2. " etc at start of line
    clause_re = re.compile(r"(?m)^(\d{1,2}\.\s)")
    matches = list(clause_re.finditer(text))
    if len(matches) < 2:
        # just hard-split by length as last resort
        out = []
        for i in range(0, len(text), MAX_CHUNK_CHARS):
            out.append((f"{label} (phần {i // MAX_CHUNK_CHARS + 1})", text[i:i + MAX_CHUNK_CHARS]))
        return out
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((label, text[start:end].strip()))
    return out


def main():
    files = sorted(glob.glob(os.path.join(CONTEXTS_DIR, "*.json")))
    print(f"Found {len(files)} context files")

    n_chunks = 0
    n_docs_no_article = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception as e:
                print("skip (parse error):", fp, e)
                continue

            doc_id = doc.get("id")
            doc_name = doc.get("name", "")
            link = doc.get("link", "")
            passage = doc.get("passage", "") or ""
            passage = clean_text(passage)
            if not passage:
                continue

            articles = split_into_articles(passage)
            if articles is None:
                n_docs_no_article += 1
                articles = sliding_window_fallback(passage)

            for label, text in articles:
                for sub_label, sub_text in maybe_subsplit_article(label, text):
                    if not sub_text.strip():
                        continue
                    chunk = {
                        "chunk_id": f"{doc_id}_{n_chunks}",
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "link": link,
                        "article": sub_label,
                        "text": sub_text,
                    }
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1

    print(f"Wrote {n_chunks} chunks to {OUT_PATH}")
    print(f"Documents with no detectable 'Điều' marker (used sliding window): {n_docs_no_article}")


if __name__ == "__main__":
    main()
