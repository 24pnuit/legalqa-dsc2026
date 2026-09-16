# Pipeline baseline (BM25 + extractive) — Task 2 LegalQA

Chạy độc lập, không phụ thuộc vào retrieval/generator "xịn" của Nhung/Nhi.
Khi có bản thật, chỉ cần thay bước 02 (BM25 → embedding/hybrid) và bước 03
(extractive → LLM generation), giữ nguyên 01/04/05.

## Thứ tự chạy
1. `01_chunk_corpus.py`   — đọc 8532 file trong selected-contexts, làm sạch
   whitespace, chunk theo "Điều" (fallback: sliding window nếu không có
   marker "Điều"), ghi ra `data/chunks.jsonl`.
2. `02_build_bm25.py`     — build BM25 index (dùng thư viện `bm25s`, nhanh
   và ít tốn RAM hơn `rank_bm25` với corpus lớn ~375k chunks). Lưu vào
   `data/bm25_index/`.
3. `03_generate_extractive.py` — với mỗi câu hỏi trong `public-official.json`,
   lấy top-3 chunk BM25, ghép thành answer dạng
   "Theo <tên văn bản>, <Điều>: <nội dung>", cắt bớt còn tối đa 350 từ
   (~median độ dài đáp án train). Ghi `data/submission.json`.
4. `04_validate_submission.py` — kiểm tra đủ 1000/1000 ID, không null,
   answer là string không rỗng, JSON hợp lệ UTF-8.
5. `05_make_submission_zip.py` — đóng gói `submission.zip` chứa đúng 1 file
   `submission.json` bên trong, đúng format BTC yêu cầu.

## Lưu ý
- Đây là baseline TỐI THIỂU (extractive thuần, không có LLM, không rerank,
  không hybrid) — điểm sẽ thấp, mục đích chỉ để có 1 bản nộp end-to-end
  đúng hạn Ngày 3, không phải bản cuối cùng.
- Khi Nhung có BM25/dense/hybrid tốt hơn: thay nội dung `02_build_bm25.py`,
  giữ nguyên format `chunks.jsonl` (đã thống nhất field:
  chunk_id, document_id, document_name, link, article, text).
- Khi Nhi có generator LLM thật: thay hàm `build_extractive_answer()` trong
  `03_generate_extractive.py` bằng lời gọi model, input vẫn là list các
  chunk đã retrieve — không cần sửa gì ở 01/04/05.
