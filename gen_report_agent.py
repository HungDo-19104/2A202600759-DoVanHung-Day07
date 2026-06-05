from pathlib import Path
from src import (
    Document, RecursiveChunker, EmbeddingStore, KnowledgeBaseAgent, LocalEmbedder,
)

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
    ("Lecture/Seminar",
     "Theo Quy định ban hành kèm theo Quyết định số 317/QĐ-ĐHKTQD, mô hình giảng dạy Lecture/Seminar được tổ chức như thế nào và giảng viên dạy lớp Lecture cần đáp ứng những điều kiện gì?"),
    ("Buộc thôi học",
     "Sinh viên đại học chính quy sẽ bị buộc thôi học trong những trường hợp nào theo Quy chế đào tạo trình độ đại học mới nhất (Quyết định số 712/QĐ-ĐHKTQD)?"),
    ("Quy trình học vụ",
     "Nếu một sinh viên muốn xin thôi học hoặc nghỉ học tạm thời, quy trình thực hiện trên tài khoản sinh viên và các bước tiếp theo diễn ra như thế nào?"),
    ("Công nhận KQHT",
     "Các điều kiện và tiêu chí để sinh viên được công nhận kết quả học tập và chuyển đổi tín chỉ sang các học phần trong chương trình đào tạo tại Đại học Kinh tế Quốc dân là gì?"),
    ("Thực tập & Khóa luận",
     "Quy trình tổ chức học phần 'Khóa luận tốt nghiệp' đối với sinh viên bao gồm các bước nào và cách tính điểm khóa luận được quy định ra sao?"),
]

def demo_llm(prompt: str) -> str:
    context_part = prompt.split("Context:")[1].split("Question:")[0].strip()[:200] if "Context:" in prompt else ""
    return f"[Agent Answer based on context] Found relevant information about the query. Key context: {context_part}..."

embedder = LocalEmbedder()

docs = []
for fname in DOC_FILES:
    path = BASE / fname
    content = path.read_text(encoding="utf-8")
    docs.append(Document(id=path.stem, content=content, metadata={"source": fname}))

chunker = RecursiveChunker(chunk_size=500)
chunked_docs = []
for doc in docs:
    for i, chunk in enumerate(chunker.chunk(doc.content)):
        chunked_docs.append(Document(
            id=f"{doc.id}_chunk_{i:03d}", content=chunk,
            metadata={**doc.metadata, "doc_id": doc.id},
        ))

store = EmbeddingStore(collection_name="report", embedding_fn=embedder)
store.add_documents(chunked_docs)
agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

print("QUERY RESULTS FOR REPORT")
print("=" * 70)
for short_name, query in QUERIES:
    results = store.search(query, top_k=3)
    agent_answer = agent.answer(query, top_k=3)
    print(f"\n--- {short_name} ---")
    print(f"Query: {query[:100]}...")
    for i, r in enumerate(results, 1):
        preview = r["content"][:150].replace("\n", " ")
        print(f"  #{i} score={r['score']:.4f} [{r['metadata']['source']}]: {preview}...")
    print(f"  Agent: {agent_answer[:150]}...")
