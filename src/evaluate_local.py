import argparse
import json
import os
from pathlib import Path
import nltk
from nltk.translate.meteor_score import meteor_score
import numpy as np
from rouge_score import rouge_scorer

# Tải gói phụ trợ cho METEOR nếu chưa có
try:
  nltk.data.find('corpora/wordnet')
except LookupError:
  nltk.download('wordnet', quiet=True)
  nltk.download('omw-1.4', quiet=True)


def read_json(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    return json.load(f)


def eval_qa(y_pred_raw, y_true_raw):
  """Hàm tính điểm chuẩn xác 100% theo mã nguồn scoring.py của BTC UIT DSC 2026."""
  rouge_scoring = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

  # BTC không dùng word segmentation
  def build_in_tokenizer(string_sent):
    return string_sent

  # Chuẩn hóa format y_pred: {qid: answer_text}
  y_pred = {}
  for k, v in y_pred_raw.items():
    if isinstance(v, dict) and 'answer' in v:
      y_pred[k] = v['answer']
    else:
      y_pred[k] = str(v)

  # Chuẩn hóa format y_true
  y_true = {}
  for k, v in y_true_raw.items():
    if isinstance(v, dict) and 'answer' in v:
      y_true[k] = v['answer']
    else:
      y_true[k] = str(v)

  ids_preds = list(y_pred.keys())
  ids_truth = list(y_true.keys())

  common_ids = [qid for qid in ids_truth if qid in y_pred]

  if len(common_ids) != len(ids_truth):
    print(
        f'[CẢNH BÁO] Số mẫu dự đoán ({len(common_ids)}) không khớp hoàn toàn'
        f' với tập mẫu ({len(ids_truth)})!'
    )

  meteor_list = []
  rouge_list = []

  for k in common_ids:
    ref_tokens = build_in_tokenizer(str(y_true[k])).split()
    pred_tokens = build_in_tokenizer(str(y_pred[k])).split()

    # 1. METEOR
    m_val = meteor_score([ref_tokens], pred_tokens) if pred_tokens else 0.0
    meteor_list.append(m_val)

    # 2. ROUGE-L
    r_val = rouge_scoring.score(
        build_in_tokenizer(str(y_true[k])), build_in_tokenizer(str(y_pred[k]))
    )['rougeL'].fmeasure
    rouge_list.append(r_val)

  meteor_mean = float(np.mean(meteor_list)) if meteor_list else 0.0
  rouge_mean = float(np.mean(rouge_list)) if rouge_list else 0.0

  return {'meteor': meteor_mean, 'rouge': rouge_mean, 'evaluated': len(common_ids)}


def main():
  parser = argparse.ArgumentParser(
      description='Đánh giá Local Task 2 LegalQA chuẩn BTC'
  )
  parser.add_argument(
      '--ref',
      type=str,
      default='data/splits/val_reference.json',
      help='Đường dẫn file tham chiếu',
  )
  parser.add_argument(
      '--pred',
      type=str,
      required=True,
      help='Đường dẫn file dự đoán (submission.json)',
  )
  args = parser.parse_args()

  truth = read_json(args.ref)
  preds = read_json(args.pred)

  res = eval_qa(preds, truth)

  print('\n' + '=' * 45)
  print('      KẾT QUẢ ĐÁNH GIÁ NỘI BỘ (LOCAL)')
  print('=' * 45)
  print(f'Số lượng câu hỏi khớp : {res["evaluated"]}')
  print(f"METEOR (Độ đo chính)  : {res['meteor']:.6f}")
  print(f"ROUGE-L (Độ đo phụ)   : {res['rouge']:.6f}")
  print('=' * 45 + '\n')


if __name__ == '__main__':
  main()