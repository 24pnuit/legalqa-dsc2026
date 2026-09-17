"""Build a BM25S index from the processed LegalQA chunk corpus.

The input is read line-by-line. Only compact integer token arrays are retained
during the build; full chunk dictionaries are streamed again when BM25S saves
``corpus.jsonl``. This preserves ``text`` and all metadata for later stages
without also holding the 1+ GiB JSONL corpus in memory.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
from array import array
from pathlib import Path

import bm25s

from bm25_utils import TOKEN_PATTERN, iter_tokens, tokenize_queries


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("data/bm25_index")
REQUIRED_FIELDS = {
    "chunk_id",
    "document_id",
    "document_name",
    "source_url",
    "text",
    "search_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM25S from chunks.jsonl, indexing search_text."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--smoke-query",
        default="Cơ sở sản xuất tôm sú giống có quyền và nghĩa vụ gì?",
        help="Query used to verify save/load and corpus-position mapping.",
    )
    return parser.parse_args()


def read_and_tokenize(chunks_path: Path):
    """Validate and tokenize JSONL while calculating its exact SHA-256."""
    token_ids: list[array] = []
    vocabulary: dict[str, int] = {}
    chunk_ids: set[str] = set()
    fields: set[str] = set()
    digest = hashlib.sha256()
    token_count = 0

    with chunks_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                chunk = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid UTF-8/JSON at line {line_number}: {exc}") from exc

            if not isinstance(chunk, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            missing = REQUIRED_FIELDS.difference(chunk)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {', '.join(sorted(missing))}"
                )

            chunk_id = chunk["chunk_id"]
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Line {line_number} has an invalid chunk_id")
            if chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk_id at line {line_number}: {chunk_id}")
            chunk_ids.add(chunk_id)

            for name in ("text", "search_text"):
                if not isinstance(chunk[name], str) or not chunk[name].strip():
                    raise ValueError(f"Line {line_number} has invalid {name}")

            document_tokens = array("I")
            for token in iter_tokens(chunk["search_text"]):
                token_id = vocabulary.get(token)
                if token_id is None:
                    token_id = len(vocabulary)
                    if token_id >= 2**32:
                        raise OverflowError("BM25 vocabulary exceeds uint32 capacity")
                    vocabulary[token] = token_id
                document_tokens.append(token_id)

            if not document_tokens:
                raise ValueError(f"Line {line_number} search_text produces no tokens")
            token_count += len(document_tokens)
            token_ids.append(document_tokens)
            fields.update(chunk)

            if line_number % 25_000 == 0:
                print(
                    f"  tokenized {line_number:,} chunks | "
                    f"tokens={token_count:,} | vocab={len(vocabulary):,}",
                    flush=True,
                )

    if not token_ids:
        raise ValueError(f"No chunks found in {chunks_path}")
    return token_ids, vocabulary, digest.hexdigest(), token_count, sorted(fields)


def iter_chunks(chunks_path: Path):
    """Yield full records so BM25S persists text and every metadata field."""
    with chunks_path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_peak_rss_bytes() -> int | None:
    """Return peak resident/working-set bytes on Windows or POSIX."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        if get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
        return None

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if os.uname().sysname == "Darwin" else peak * 1024)
    except (AttributeError, ImportError):
        return None


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def replace_directory(staged_dir: Path, target_dir: Path) -> None:
    """Replace an existing index only after the new one is complete."""
    backup_dir = None
    if target_dir.exists():
        backup_dir = target_dir.with_name(f".{target_dir.name}.old-{os.getpid()}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        target_dir.rename(backup_dir)
    try:
        staged_dir.rename(target_dir)
    except Exception:
        if backup_dir is not None and not target_dir.exists():
            backup_dir.rename(target_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks.resolve()
    index_dir = args.index_dir.resolve()
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk corpus not found: {chunks_path}")

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{index_dir.name}.building-", dir=index_dir.parent)
    )
    started = time.perf_counter()

    try:
        print(f"Reading and tokenizing search_text from {chunks_path}", flush=True)
        step_started = time.perf_counter()
        token_ids, vocabulary, chunks_sha256, token_count, corpus_fields = (
            read_and_tokenize(chunks_path)
        )
        tokenization_seconds = time.perf_counter() - step_started
        chunk_count = len(token_ids)
        print(
            f"Tokenized {chunk_count:,} chunks / {token_count:,} tokens / "
            f"{len(vocabulary):,} terms in {tokenization_seconds:.1f}s",
            flush=True,
        )

        step_started = time.perf_counter()
        retriever = bm25s.BM25(method="lucene")
        tokenized = bm25s.tokenization.Tokenized(ids=token_ids, vocab=vocabulary)
        retriever.index(tokenized, show_progress=True)
        index_seconds = time.perf_counter() - step_started
        print(f"Built BM25 index in {index_seconds:.1f}s", flush=True)

        # Release the largest build-only objects before streaming the full corpus.
        del tokenized, token_ids, vocabulary

        step_started = time.perf_counter()
        retriever.save(staged_dir, corpus=iter_chunks(chunks_path), show_progress=True)
        save_seconds = time.perf_counter() - step_started
        print(f"Saved index and full chunk records in {save_seconds:.1f}s", flush=True)

        # Verify persisted position -> record mapping through the same load path used
        # by downstream consumers. On Windows, JsonlCorpus holds an mmap/file handle
        # that must be closed before the staged directory can be renamed or removed.
        loaded = None
        try:
            loaded = bm25s.BM25.load(
                staged_dir, load_corpus=True, mmap=True, show_progress=False
            )
            query_tokens = tokenize_queries([args.smoke_query])
            results, scores = loaded.retrieve(
                query_tokens, corpus=loaded.corpus, k=min(3, chunk_count), show_progress=False
            )
            smoke_results = []
            for record, score in zip(results[0], scores[0]):
                if not isinstance(record.get("text"), str) or "search_text" not in record:
                    raise RuntimeError("Saved corpus lost text/search_text metadata")
                smoke_results.append(
                    {
                        "chunk_id": record["chunk_id"],
                        "document_id": record["document_id"],
                        "score": float(score),
                    }
                )
        finally:
            if loaded is not None:
                try:
                    corpus_obj = getattr(loaded, "corpus", None)
                    if corpus_obj is not None and hasattr(corpus_obj, "close"):
                        corpus_obj.close()
                except Exception:
                    pass
                del loaded
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass

        elapsed = time.perf_counter() - started
        manifest = {
            "format_version": 1,
            "source": {
                "chunks_path": str(chunks_path),
                "sha256": chunks_sha256,
                "size_bytes": chunks_path.stat().st_size,
                "chunk_count": chunk_count,
                "fields": corpus_fields,
            },
            "indexed_field": "search_text",
            "stored_corpus": "corpus.jsonl",
            "stored_fields": corpus_fields,
            "tokenizer": {
                "unicode_normalization": "NFC",
                "lowercase": True,
                "token_pattern": TOKEN_PATTERN,
                "stopwords": None,
            },
            "bm25": {
                "library": "bm25s",
                "version": importlib.metadata.version("bm25s"),
                "method": retriever.method,
                "k1": retriever.k1,
                "b": retriever.b,
                "delta": retriever.delta,
                "vocabulary_size": len(retriever.vocab_dict),
                "token_count": token_count,
            },
            "build": {
                "git_commit": get_git_commit(),
                "python": os.sys.version.split()[0],
                "tokenization_seconds": round(tokenization_seconds, 3),
                "index_seconds": round(index_seconds, 3),
                "save_seconds": round(save_seconds, 3),
                "total_seconds": round(elapsed, 3),
                "peak_rss_bytes": get_peak_rss_bytes(),
            },
            "smoke_test": {"query": args.smoke_query, "top_results": smoke_results},
        }
        (staged_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["build"]["index_size_bytes"] = directory_size(staged_dir)
        (staged_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        replace_directory(staged_dir, index_dir)
        print(f"Index ready: {index_dir}", flush=True)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        # Ensure any mmap handles are released before attempting cleanup on Windows.
        try:
            if "loaded" in locals() and loaded is not None:
                corpus_obj = getattr(loaded, "corpus", None)
                if corpus_obj is not None and hasattr(corpus_obj, "close"):
                    corpus_obj.close()
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        if staged_dir.exists():
            # On Windows, retry with onerror to handle lingering handles.
            def _on_rm_error(func, path, exc_info):
                try:
                    import time

                    time.sleep(0.2)
                    func(path)
                except Exception:
                    pass

            try:
                shutil.rmtree(staged_dir, onerror=_on_rm_error)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
