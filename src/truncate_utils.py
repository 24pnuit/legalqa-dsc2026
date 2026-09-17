"""
Cắt bớt văn bản pháp luật tại ranh giới câu hoặc ranh giới Khoản, thay vì cắt
cứng theo số từ (word count) — tránh cắt giữa chừng một điều khoản/số tiền/tỷ lệ.

Chiến lược:
  1. Tìm vị trí ký tự tương ứng với `max_words` từ (ước lượng).
  2. Tìm tất cả "điểm cắt an toàn" trong văn bản:
       - Ngay sau dấu kết câu (. ; :) theo sau bởi khoảng trắng/xuống dòng.
       - Ngay TRƯỚC một khoản mới bắt đầu (dòng dạng "1. ", "2. " ở đầu dòng)
         — cắt trước khi khoản mới bắt đầu, để không cắt dở khoản đó.
  3. Chọn điểm cắt an toàn gần nhất mà không vượt quá vị trí ước lượng.
  4. Nếu không tìm được điểm cắt an toàn nào (văn bản không có dấu câu rõ ràng,
     hiếm gặp) -> fallback cắt cứng theo từ như cũ (an toàn hơn là crash).
"""
import re

SENTENCE_END_RE = re.compile(r'[.;:]["\')]?\s')
KHOAN_START_RE = re.compile(r'\n(?=\d{1,2}[a-z]?\.\s)')
# Marker số khoản ở đầu dòng, vd "1. ", "2a. " — cần loại khỏi SENTENCE_END_RE vì
# "số + chấm + khoảng trắng" trông giống dấu kết câu nhưng thực ra là số thứ tự.
KHOAN_MARKER_RE = re.compile(r'(?m)^(\d{1,2}[a-z]?)\.\s')


def _char_offset_for_word_count(text, max_words):
    """Ước lượng vị trí ký tự tương ứng với max_words từ đầu tiên."""
    words = text.split()
    if len(words) <= max_words:
        return len(text)
    # ghép lại max_words từ đầu để lấy độ dài ký tự tương ứng
    prefix = " ".join(words[:max_words])
    return len(prefix)


def find_safe_cut_points(text):
    """Trả về list vị trí ký tự có thể cắt an toàn (đã sort tăng dần)."""
    # các khoảng [start,end) là marker số khoản "1. ", "2a. " -> loại khỏi sentence-end
    khoan_marker_spans = [(m.start(), m.end()) for m in KHOAN_MARKER_RE.finditer(text)]

    def inside_khoan_marker(pos):
        return any(s <= pos < e for s, e in khoan_marker_spans)

    points = set()
    for m in SENTENCE_END_RE.finditer(text):
        if inside_khoan_marker(m.start()):
            continue  # đây là số thứ tự khoản ("3. "), không phải dấu kết câu thật
        points.add(m.end())
    for m in KHOAN_START_RE.finditer(text):
        points.add(m.start())
    points.add(0)
    return sorted(points)


def truncate_at_boundary(text, max_words, min_ratio=0.5):
    """
    Cắt `text` còn tối đa khoảng max_words từ, tại ranh giới câu/khoản gần nhất.

    min_ratio: nếu điểm cắt an toàn gần nhất làm mất quá nhiều nội dung (dưới
    min_ratio * max_words từ so với dự kiến), coi như không đáng tin cậy và
    dùng fallback cắt cứng theo từ (tránh trả lời cụt lủn chỉ còn vài chữ).
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    target_offset = _char_offset_for_word_count(text, max_words)
    safe_points = find_safe_cut_points(text)

    candidates = [p for p in safe_points if p <= target_offset]
    if not candidates:
        # fallback: cắt cứng theo từ (không có điểm cắt an toàn nào trước target)
        return " ".join(words[:max_words]).strip()

    best_cut = max(candidates)
    cut_word_count = len(text[:best_cut].split())

    if cut_word_count < max_words * min_ratio:
        # điểm cắt an toàn quá xa target -> fallback cắt cứng để không bị cụt quá nhiều
        return " ".join(words[:max_words]).strip()

    return text[:best_cut].strip()


if __name__ == "__main__":
    # --- self-test nhanh ---
    sample = (
        "Điều 3. Các Ông (Bà) Chánh Văn phòng, Chánh Thanh tra, Cục trưởng Cục "
        "Nhà giáo và Cán bộ quản lý cơ sở giáo dục, Thủ trưởng các đơn vị có "
        "liên quan thuộc Bộ Giáo dục và Đào tạo, Chủ tịch Uỷ ban nhân dân tỉnh, "
        "thành phố trực thuộc Trung ương, Giám đốc các sở Giáo dục và Đào tạo "
        "chịu trách nhiệm thi hành Thông tư này.\n"
        "1. Thông tư có hiệu lực kể từ ngày ký.\n"
        "2. Các quy định trước đây trái với Thông tư này đều bị bãi bỏ.\n"
        "3. Trong quá trình thực hiện nếu có vướng mắc đề nghị phản ánh kịp thời."
    )
    for mw in [15, 25, 40, 100]:
        out = truncate_at_boundary(sample, mw)
        print(f"max_words={mw:4d} -> {len(out.split())} từ thực tế | {out!r}")
