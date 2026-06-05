from pathlib import Path
from src import (
    ChunkingStrategyComparator, compute_similarity, LocalEmbedder,
    RecursiveChunker, FixedSizeChunker, SentenceChunker,
)

print("=" * 60)
print("CHUNKING COMPARISON")
print("=" * 60)

files = [
    "q_317_0642023_day_va_hoc_theo_mo_hinh_lecture_seminar_tai_hktqd.md",
    "quyet_inh_so_501_ngay_0752026_vv_ban_hanh_quy_trinh_to_chuc_chuyen_e_thuc_te_oi_voi_sv_h_ktqd.md",
]

for fname in files:
    text = Path(fname).read_text(encoding="utf-8")
    result = ChunkingStrategyComparator().compare(text, chunk_size=200)
    print(f"\n--- {fname[:40]} ---")
    for s, v in result.items():
        print(f"  {s:15s}: count={v['count']:3d}, avg={v['avg_length']:7.1f}")

print("\n" + "=" * 60)
print("SIMILARITY PREDICTIONS (using LocalEmbedder)")
print("=" * 60)

pairs = [
    ("Mô hình Lecture/Seminar kết hợp giữa lớp Lecture và lớp Seminar",
     "Lớp Lecture có quy mô không quá 300 sinh viên, lớp Seminar từ 20-30 sinh viên"),
    ("Sinh viên bị buộc thôi học nếu cảnh báo học tập vượt quá 2 lần",
     "Sinh viên đăng ký học phần trên phần mềm quản lý đào tạo"),
    ("Khóa luận tốt nghiệp có thời gian tối thiểu 10 tuần",
     "Thực tập tốt nghiệp là hoạt động ở giai đoạn cuối khóa học"),
    ("Điểm chuyên cần có trọng số 10% đánh giá quá trình học",
     "Giảng viên hướng dẫn chấm điểm khóa luận theo thang 10"),
    ("Mô hình Lecture/Seminar kết hợp giữa lớp Lecture và lớp Seminar",
     "Quy trình tổ chức Chuyên đề thực tế, Thực tập giữa khóa, Kiến tập"),
]

e = LocalEmbedder()
for i, (a, b) in enumerate(pairs, 1):
    score = compute_similarity(e(a), e(b))
    print(f"\nPair {i}: {score:.4f}")
    print(f"  A: {a}")
    print(f"  B: {b}")
