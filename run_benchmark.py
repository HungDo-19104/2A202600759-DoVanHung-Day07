import os
from pathlib import Path
from src import (
    Document, RecursiveChunker, EmbeddingStore,
    ChunkingStrategyComparator, compute_similarity, LocalEmbedder,
)

embedder = LocalEmbedder()

BASE = Path(__file__).parent

DOC_FILES = [
    "q_317_0642023_day_va_hoc_theo_mo_hinh_lecture_seminar_tai_hktqd.md",
    "q_712_0362025_quy_che_ao_tao_ai_hoc.md",
    "q_715_0362025_quy_che_tuyen_sinh_ai_hoc.md",
    "q_so_110_quy_che_xet_cap_hoc_bong_oi_voi_sinh_vien_he_chinh_quy.md",
    "qd_ban_hanh_quy_che_danh_gia_drl_2018_scan.md",
    "quyet_inh_so_501_ngay_0752026_vv_ban_hanh_quy_trinh_to_chuc_chuyen_e_thuc_te_oi_voi_sv_h_ktqd.md",
]

QUERIES = [
    (
        "Mô hình Lecture/Seminar",
        "Theo Quy định ban hành kèm theo Quyết định số 317/QĐ-ĐHKTQD, "
        "mô hình giảng dạy Lecture/Seminar được tổ chức như thế nào "
        "và giảng viên dạy lớp Lecture cần đáp ứng những điều kiện gì?"
    ),
    (
        "Xử lý học tập",
        "Sinh viên đại học chính quy sẽ bị buộc thôi học trong những trường hợp nào "
        "theo Quy chế đào tạo trình độ đại học mới nhất (Quyết định số 712/QĐ-ĐHKTQD)?"
    ),
    (
        "Quy trình học vụ",
        "Nếu một sinh viên muốn xin thôi học hoặc nghỉ học tạm thời, "
        "quy trình thực hiện trên tài khoản sinh viên và các bước tiếp theo diễn ra như thế nào?"
    ),
    (
        "Công nhận kết quả học tập",
        "Các điều kiện và tiêu chí để sinh viên được công nhận kết quả học tập "
        "và chuyển đổi tín chỉ sang các học phần trong chương trình đào tạo "
        "tại Đại học Kinh tế Quốc dân là gì?"
    ),
    (
        "Thực tập & Khóa luận",
        "Quy trình tổ chức học phần 'Khóa luận tốt nghiệp' đối với sinh viên "
        "bao gồm các bước nào và cách tính điểm khóa luận được quy định ra sao?"
    ),
]


def load_docs():
    docs = []
    for fname in DOC_FILES:
        path = BASE / fname
        if not path.exists():
            print(f"  SKIP: {fname} not found")
            continue
        content = path.read_text(encoding="utf-8")
        doc_id = path.stem
        docs.append(Document(
            id=doc_id,
            content=content,
            metadata={"source": fname, "type": "policy"},
        ))
    return docs


def main():
    print("=" * 70)
    print("BENCHMARK: Recursive Character Splitting — 5 Queries")
    print("=" * 70)

    # 1. Load documents
    docs = load_docs()
    print(f"\nLoaded {len(docs)} documents:")
    for d in docs:
        print(f"  - {d.id} ({len(d.content)} chars)")

    # 2. Chunk with RecursiveChunker
    chunker = RecursiveChunker(chunk_size=500)
    chunked_docs = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                id=f"{doc.id}_chunk_{i:03d}",
                content=chunk,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i},
            ))
    print(f"\nRecursiveChunker tạo {len(chunked_docs)} chunks từ {len(docs)} documents")

    # 3. Store in EmbeddingStore
    store = EmbeddingStore(collection_name="benchmark", embedding_fn=embedder)
    store.add_documents(chunked_docs)
    print(f"EmbeddingStore size: {store.get_collection_size()}")

    # 4. Run queries
    print("\n" + "=" * 70)
    print("KẾT QUẢ SEARCH TOP-3 CHO 5 QUERIES")
    print("=" * 70)

    for short_name, query in QUERIES:
        print(f"\n{'─' * 70}")
        print(f"QUERY: {short_name}")
        print(f"  {query[:120]}...")
        print(f"{'─' * 70}")

        results = store.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "?")
            preview = r["content"][:200].replace("\n", " ")
            print(f"\n  #{i}  score={r['score']:.4f}  [{source}]")
            print(f"       {preview}...")

    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH! Tất cả 5 queries đã được chạy với RecursiveChunker.")


if __name__ == "__main__":
    main()
