# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Văn Hùng
**Nhóm:** A3
**Ngày:** 05/06/2026
**Thành viên:** Lê Hoàng Nam, Lê Trần Quốc Bảo, Đỗ Văn Hùng, Bùi Văn Thái, Đặng Ngọc Bách

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai câu/text chunks có cosine similarity cao nghĩa là vector biểu diễn của chúng gần nhau trong không gian embedding, tức là chúng có nội dung ngữ nghĩa tương đồng.

**Ví dụ HIGH similarity:**
- Sentence A: Mô hình Lecture/Seminar kết hợp giữa lớp Lecture và lớp Seminar
- Sentence B: Lớp Lecture có quy mô không quá 300 sinh viên, lớp Seminar từ 20-30 sinh viên
- Tại sao tương đồng: Cùng nói về mô hình Lecture/Seminar, cùng chủ đề tổ chức giảng dạy

**Ví dụ LOW similarity:**
- Sentence A: Khóa luận tốt nghiệp có thời gian tối thiểu 10 tuần
- Sentence B: Sinh viên bị buộc thôi học nếu cảnh báo học tập vượt quá 2 lần
- Tại sao khác: Một câu nói về quy trình tốt nghiệp, câu kia về kỷ luật học tập

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc giữa hai vector (hướng ngữ nghĩa), không bị ảnh hưởng bởi độ dài vector (norm), giúp so sánh các văn bản có độ dài khác nhau một cách công bằng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks** (tăng 2 chunks). Overlap nhiều hơn giúp giữ ngữ cảnh giữa các chunk, tránh mất thông tin ở ranh giới.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Chính sách & Quy chế đào tạo Đại học Kinh tế Quốc dân

**Tại sao nhóm chọn domain này?**
> Nhóm thống nhất chọn domain này vì các văn bản quy chế đại học có đặc thù là rất dài, cấu trúc phân cấp rõ ràng (theo Quyết định, Chương, Điều, Khoản) nhưng ngôn ngữ hành chính phức tạp. Đây là bài toán thực tế hữu ích giúp sinh viên tra cứu thông tin chính xác, đồng thời là bộ dữ liệu lý tưởng để kiểm thử sức mạnh của RAG, các chiến lược chia nhỏ văn bản (chunking) và lọc siêu dữ liệu (metadata filtering).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|----|--------------|-------|----------|-----------------|
| 1 | Quyết định 317: Dạy và học Lecture/Seminar | NEU | ~7,078 | category=dao_tao, year=2023, doc_type=quy_dinh |
| 2 | Quyết định 712: Quy chế đào tạo ĐH | NEU | ~225,538 | category=dao_tao, year=2025, doc_type=quy_che |
| 3 | Quyết định 715: Quy chế tuyển sinh ĐH | NEU | ~94,329 | category=tuyen_sinh, year=2025, doc_type=quy_che |
| 4 | Quyết định 110: Quy chế học bổng | NEU | ~22,354 | category=hoc_bong, year=2022, doc_type=quy_che |
| 5 | Quyết định: Quy chế đánh giá ĐRL | NEU | ~55,499 | category=diem_ren_luyen, year=2018, doc_type=quy_che |
| 6 | Quyết định 501: Quy trình thực tập, KLTN | NEU | ~33,359 | category=thuc_tap, year=2026, doc_type=quy_trinh |

### Metadata Schema Thống Nhất

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id / source | string | q_712_0362025... | Xác định văn bản gốc, hỗ trợ filter & delete |
| category | string | dao_tao, tuyen_sinh, hoc_bong | Lọc thu hẹp không gian tìm kiếm theo lĩnh vực |
| year | int/string | 2025, 2026 | Ưu tiên văn bản mới nhất có hiệu lực |
| doc_type | string | quy_che, quy_dinh, quy_trinh | Phân biệt loại văn bản |
| parent_id | string | UUID của parent chunk | Liên kết child chunk với parent context |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu với chunk_size=200:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| q_317 (Lecture/Seminar) | fixed_size | 29 | 193.4 | Trung bình — cắt cứng |
| q_317 (Lecture/Seminar) | by_sentences | 15 | 371.6 | Tốt — giữ nguyên câu |
| q_317 (Lecture/Seminar) | recursive | 37 | 149.7 | Tốt — giữ cấu trúc văn bản |
| Quyết định 501 | fixed_size | 134 | 198.7 | Trung bình |
| Quyết định 501 | by_sentences | 67 | 388.4 | Tốt |
| Quyết định 501 | recursive | 200 | 143.5 | Tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker (Recursive Character Splitting)

**Mô tả cách hoạt động:**
> Chunker thử tách text bằng các separator theo thứ tự ưu tiên: `\n\n` (ngắt đoạn), `\n` (xuống dòng), `. ` (hết câu), ` ` (khoảng trắng), và cuối cùng là split ký tự. Nếu một piece sau khi tách vẫn còn lớn hơn chunk_size, nó được đệ quy xử lý với separator tiếp theo. Các chunk nhỏ cạnh nhau được gộp lại nếu tổng kích thước không vượt quá chunk_size.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản quy chế có cấu trúc phân cấp rõ ràng (Chương > Điều > Khoản), RecursiveChunker tận dụng các separator tự nhiên (`\n\n` giữa các Điều, `\n` giữa các Khoản) để tạo chunks có ý nghĩa, giữ được ngữ cảnh tốt hơn FixedSizeChunker.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| q_317 | fixed_size (baseline) | 29 | 193.4 | Có thể mất ngữ cảnh ở ranh giới |
| q_317 | **recursive (của tôi)** | 37 | 149.7 | Chunks giữ cấu trúc điều khoản |
| Quyết định 501 | fixed_size (baseline) | 134 | 198.7 | Cắt cứng, mất ngữ cảnh |
| Quyết định 501 | **recursive (của tôi)** | 200 | 143.5 | Giữ nguyên đoạn văn bản |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| I (Hung) | RecursiveChunker | 8 | Giữ cấu trúc văn bản | Nhieu chunks hon, co the dut doan danh sach dai |
| Le Hoang Nam | Parent-Child / Small-to-big | 9 | Can bang chinh xac va boi canh | Phuc tap hon khi cai dat |
| Le T. Quoc Bao | Semantic Chunker | 8 | Ranh gioi chunk khop chuyen y | Index cham, ton chi phi API |
| Bui Van Thai | Markdown Header Chunker | 8 | Bao toan tron ven mot Dieu luat | Phu thuoc format Markdown |
| Dang Ngoc Bach | Fixed-size (Overlap lon) | 7 | Don gian, an toan | Du thua thong tin lap lai |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Parent-Child/Small-to-big la tot nhat. Van ban quy che can chinh xac cap do Khoan (child) nhung luon can boi canh toan bo Dieu (parent) de giai thich dung luat. Ket hop voi Metadata Filter se la phuong an toi uu.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])\s+` để tách câu dựa trên dấu câu kết thúc. Sau đó gom các câu thành nhóm theo `max_sentences_per_chunk`. Xử lý edge case: text rỗng → `[]`, số câu ≤ max → trả về 1 chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Base case: nếu độ dài text ≤ chunk_size → `[text]`. Nếu hết separator → split ký tự thuần. Thuật toán: thử separator ưu tiên cao nhất, nếu chỉ có 1 piece → đệ quy với separator tiếp theo. Các piece được gom vào buffer, nếu vượt chunk_size thì đệ quy xử lý buffer với separator tiếp theo. Cuối cùng gộp các chunk nhỏ liền kề nếu tổng ≤ chunk_size.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: mỗi Document được `_make_record` tạo dict gồm id, content, metadata, embedding, sau đó append vào `self._store`. `search`: embed query → dot product với tất cả stored embeddings → sort descending → trả về top_k kèm score và content.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: pre-filter — duyệt `self._store`, giữ records khớp tất cả key/value trong metadata_filter, sau đó gọi `_search_records` trên filtered list. `delete_document`: list comprehension loại bỏ records có `metadata['doc_id'] == doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Search top_k chunks từ store. Build prompt với format: `[i] (source: ...) \n {content}` cho mỗi chunk, kết hợp với câu hỏi. Gọi `llm_fn(prompt)` và trả về answer.

### Test Results

```
pytest tests/ -v → 42 passed in 0.17s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Mô hình Lecture/Seminar kết hợp giữa lớp Lecture và lớp Seminar | Lớp Lecture có quy mô không quá 300 sinh viên, lớp Seminar từ 20-30 sinh viên | high | 0.6260 | Dung |
| 2 | Sinh viên bị buộc thôi học nếu cảnh báo học tập vượt quá 2 lần | Sinh viên đăng ký học phần trên phần mềm quản lý đào tạo | low | 0.6350 | Sai (cao hon du doan) |
| 3 | Khóa luận tốt nghiệp có thời gian tối thiểu 10 tuần | Thực tập tốt nghiệp là hoạt động ở giai đoạn cuối khóa học | high | 0.6625 | Dung |
| 4 | Điểm chuyên cần có trọng số 10% đánh giá quá trình học | Giảng viên hướng dẫn chấm điểm khóa luận theo thang 10 | medium | 0.6825 | Gan dung (cao hon) |
| 5 | Mô hình Lecture/Seminar kết hợp giữa lớp Lecture và lớp Seminar | Quy trình tổ chức Chuyên đề thực tế, Thực tập giữa khóa, Kiến tập | low | 0.3775 | Dung |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 bất ngờ nhất: dù hai câu nói về chủ đề khác nhau, score vẫn 0.6350. Điều này cho thấy `all-MiniLM-L6-v2` có thể đánh giá cao sự tương đồng do cùng thuộc domain "sinh viên đại học" và có chung một số từ khóa (sinh viên, học tập). Embedding không chỉ dựa trên nghĩa đen mà còn dựa trên ngữ cảnh tổng thể.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Theo Quy dinh ban hanh kem theo Quyet dinh so 317/QD-DHKTQD, mo hinh giang day Lecture/Seminar duoc to chuc nhu the nao va giang vien day lop Lecture can dap ung nhung dieu kien gi? | Lop Lecture ≤ 300 SV, Seminar 20-30 SV, moi loai chiem 50% thoi luong. Giang vien day Lecture phai co trinh do tu tien si tro len. |
| 2 | Sinh vien dai hoc chinh quy se bi buoc thoi hoc trong nhung truong hop nao theo Quy che dao tao trinh do dai hoc moi nhat (QD 712)? | Canh bao hoc tap >2 lan lien tiep hoac >3 lan khong lien tiep; nghi hoc khong ly do tron 1 hoc ky chinh; vuot thoi gian hoc toi da; vi pham nghiem trong (thi ho, bang gia). |
| 3 | Neu mot sinh vien muon xin thoi hoc hoac nghi hoc tam thoi, quy trinh thuc hien tren tai khoan sinh vien dien ra the nao? | Tra cuu huong dan tren website QLDT; lam thu tuc va theo doi xu ly tren tai khoan sinh vien; nhan quyet dinh qua email DH; danh gia hai long. |
| 4 | Cac dieu kien va tieu chi de sinh vien duoc cong nhan ket qua hoc tap va chuyen doi tin chi tai NEU la gi? | Hoi dong doi san chuan dau ra, noi dung, khoi luong. Chuyen doi toi da khong qua 50% khoi luong hoc tap toi thieu, thuc hien truoc khi tot nghiep. |
| 5 | Quy trinh to chuc hoc phan "Khoa luan tot nghiep" bao gom cac buoc nao va cach tinh diem ra sao? | Quy trinh 7 buoc (Dang ky -> Nop DS -> Giay GT -> Phan cong GVHD -> Thuc tap -> Cham -> Nhap diem). Diem = 50% GVHD + 50% Hoi dong cham. |

### Kết Quả Của Tôi (với LocalEmbedder + RecursiveChunker)

| # | Query | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |
|---|-------|----------------------|-------|-----------|-------------|
| 1 | Lecture/Seminar | Can cu Quy che dao tao... (q_317) | 0.8265 | Co | Tra loi dua tren context ve Quyet dinh 317 |
| 2 | Buoc thoi hoc | Truong hop dac biet do Giam doc... (q_712) | 0.8295 | Co (dung file) | Tra loi dua tren context tu Quy che 712 |
| 3 | Quy trinh hoc vu | Huong dan dang ky hoc vu truc tuyen... (q_712) | 0.7739 | Co | Tra loi dua tren quy trinh hoc vu |
| 4 | Cong nhan KQHT | Cong nhan ket qua hoc tap cua nguoi hoc... (q_712) | 0.8605 | Co | Tra loi dua tren Dieu 16 Quy che 712 |
| 5 | Thuc tap & KLTN | Quy trinh cham Khoa luan tot nghiep... (q_501) | 0.8642 | Co | Tra loi dua tren Quyet dinh 501 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tu Le Hoang Nam (Parent-Child): toi hoc duoc cach ket hop child chunk de lay chinh xac va parent chunk de lay boi canh. Tu Dang Ngoc Bach: phat hien ChromaDB mac dinh dung L2 Distance thay vi Cosine Similarity, can cau hinh `hnsw:space: cosine` de tranh sai lech diem similarity.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *[Viết sau khi demo]*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm metadata chi tiết hơn (số quyết định, năm ban hành, chương số) để hỗ trợ search_with_filter tốt hơn. Ngoài ra, với các tài liệu PDF scan, cần OCR + kiểm tra chất lượng trước khi chunk.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
