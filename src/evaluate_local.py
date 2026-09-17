"""Evaluate LegalQA predictions with METEOR and ROUGE-L."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def ensure_nltk_resources(download: bool = False) -> None:
    """Require METEOR resources without network I/O during module import."""
    missing = []
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            missing.append(resource)
    if missing and download:
        for resource in missing:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(f"Failed to download NLTK resource: {resource}")
        return
    if missing:
        names = " ".join(missing)
        raise RuntimeError(
            f"Missing NLTK resources: {names}. Run "
            f"python -m nltk.downloader {names} or pass --download-nltk-data."
        )


def read_json(file_path: str | Path):
    with open(file_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def eval_qa(y_pred_raw, y_true_raw):
    """Calculate metrics using the tokenization behavior of the provided scorer."""
    rouge_scoring = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def normalize_answers(raw):
        return {
            key: value["answer"]
            if isinstance(value, dict) and "answer" in value
            else str(value)
            for key, value in raw.items()
        }

    y_pred = normalize_answers(y_pred_raw)
    y_true = normalize_answers(y_true_raw)
    if set(y_pred) != set(y_true):
        missing = sorted(set(y_true) - set(y_pred))
        extra = sorted(set(y_pred) - set(y_true))
        raise ValueError(
            "Prediction IDs must exactly match reference IDs; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    meteor_values = []
    rouge_values = []
    for qid in y_true:
        reference = str(y_true[qid])
        prediction = str(y_pred[qid])
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        meteor_values.append(
            meteor_score([reference_tokens], prediction_tokens)
            if prediction_tokens
            else 0.0
        )
        rouge_values.append(
            rouge_scoring.score(reference, prediction)["rougeL"].fmeasure
        )

    return {
        "meteor": float(np.mean(meteor_values)) if meteor_values else 0.0,
        "rouge": float(np.mean(rouge_values)) if rouge_values else 0.0,
        "evaluated": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", default="data/splits/val_reference.json", help="Reference JSON"
    )
    parser.add_argument("--pred", required=True, help="Prediction JSON")
    parser.add_argument("--out", type=Path, help="Optional metrics JSON output")
    parser.add_argument(
        "--download-nltk-data",
        action="store_true",
        help="Allow downloading WordNet/OMW if missing",
    )
    args = parser.parse_args()

    ensure_nltk_resources(download=args.download_nltk_data)
    result = eval_qa(read_json(args.pred), read_json(args.ref))

    print("\n" + "=" * 45)
    print("      KẾT QUẢ ĐÁNH GIÁ NỘI BỘ (LOCAL)")
    print("=" * 45)
    print(f'Số lượng câu hỏi khớp : {result["evaluated"]}')
    print(f'METEOR (Độ đo chính)  : {result["meteor"]:.6f}')
    print(f'ROUGE-L (Độ đo phụ)   : {result["rouge"]:.6f}')
    print("=" * 45 + "\n")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
