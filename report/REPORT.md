# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trịnh Uyên Chi
<br>**Nhóm:** B4 - C401
<br>**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector có hướng rất gần nhau trong không gian đa chiều, cho thấy nội dung giữa chúng có sự tương đồng lớn về mặt ngữ nghĩa, bất kể độ dài của văn bản.

**Ví dụ HIGH similarity:**
- Sentence A:"Tôi rất yêu thích việc lập trình ứng dụng AI."
- Sentence B: "Làm code trí tuệ nhân tạo là đam mê của tôi."
- Tại sao tương đồng: Cả hai câu đều chia sẻ cùng một trường từ vựng (lập trình/code, AI/trí tuệ nhân tạo) và biểu đạt cùng một ý nghĩa cảm xúc.

**Ví dụ LOW similarity:**
- Sentence A: "Cách làm món bún bò truyền thống."
- Sentence B: "Thị trường chứng khoán đang biến động mạnh."
- Tại sao khác nhau: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (ẩm thực và tài chính), không có sự liên quan về từ ngữ hay ngữ cảnh.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì cosine similarity tập trung vào hướng của vector thay vì độ dài. Trong văn bản, một tài liệu dài và một tài liệu ngắn có thể cùng nội dung, Euclidean distance sẽ coi chúng rất xa nhau (vì vector dài hơn), trong khi cosine vẫn nhận diện được sự tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Áp dụng công thức tính số lượng chunk:*
> $$N = \lceil \frac{L - O}{S - O} \rceil$$
> Trong đó: $L = 10,000$, $S = 500$, $O = 50$.
> $$\text{Số chunks} = \frac{10,000 - 50}{500 - 50} = \frac{9,950}{450} \approx 22.11$$
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Áp dụng công thức tính số lượng chunk:
> $$N = \lceil \frac{L - O}{S - O} \rceil$$
> 
> Thay số:
> $$N = \frac{10,000 - 100}{500 - 100} = \frac{9,900}{400} = 24.75$$
> 
> Làm tròn lên **25 chunks**
>
> Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên vì khoảng cách di chuyển của mỗi bước nhảy bị thu ngắn lại. Chúng ta muốn tăng overlap để đảm bảo các câu hoặc ngữ cảnh quan trọng không bị cắt đôi ở ranh giới giữa hai chunk, giúp mô hình Embedding hiểu nội dung liền mạch hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Xe điện VinFast

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì vừa hoàn thành cuộc thi Hackathon về trợ lý ảo AI Vivi. Việc áp dụng RAG sẽ giúp giải quyết triệt để vấn đề thông tin sai lệch về thông số kỹ thuật và chính sách pin, vốn là những nội dung đòi hỏi độ chính xác tuyệt đối.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | battery_0326 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 2839 | {"doc_code": "POL_BATT_0326", "effective_date": "2026-03-01", "policy_type": "battery", "model": "All EVs", "is_active": true} |
| 2 | charging_ev_0326 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 2944 | {"doc_code": "POL_CHAR_0326", "effective_date": "2026-03-01", "policy_type": "charging", "model": "All EVs", "is_active": true} |
| 3 | discontinued_models_0326 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 3451 | {"doc_code": "POL_DISC_0326", "effective_date": "2026-03-01", "policy_type": "support", "model": "Fadil, Lux", "is_active": true} |
| 4 | gas_to_ev_0326 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 1923 | {"doc_code": "PROMO_GAS2EV_0326", "effective_date": "2026-03-01", "policy_type": "promotion", "model": "All EVs", "is_active": true} |
| 5 | sales_0326 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 5923 | {"doc_code": "REP_SALES_0326", "effective_date": "2026-03-01", "policy_type": "sales", "model": "All", "is_active": false} |
| 6 | sales_0426 | https://vinfastauto.com/vn_vi/hop-dong-va-chinh-sach | 5895 | {"doc_code": "REP_SALES_0426", "effective_date": "2026-04-01", "policy_type": "sales", "model": "All", "is_active": true} |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_code | string | POL_BATT_0326 | Định danh để tránh lẫn lộn |
| effective_date | date | 2026-02-12 | Ưu tiên các chính sách, quyết định mới |
| policy_type | string | sales | Lọc các chính sách theo loại |
| model | string | VF3 | Filter |
| is_active | boolean | true | Đánh dấu chính sách còn hiệu lực |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 6 tài liệu (22985 ký tự):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Dữ liệu chính sách VinFast | FixedSizeChunker (`fixed_size`) | 115 | 199.9 | Không (Thường xuyên cắt ngang giữa câu) |
| Dữ liệu chính sách VinFast | SentenceChunker (`by_sentences`) | 43 | 533.6 | Trung bình (Giữ được trọn câu nhưng có thể làm đứt gãy mạch của đoạn văn) |
| Dữ liệu chính sách VinFast | RecursiveChunker (`recursive`) | 173 | 131.3 | Có (Bảo toàn tốt nhất cấu trúc phân cấp) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Chiến lược này hoạt động bằng cách chia nhỏ văn bản dựa trên một danh sách các dấu phân cách được xếp hạng ưu tiên từ lớn đến nhỏ (ví dụ: ["\n\n", "\n", ". ", " "]). Đầu tiên, thuật toán cố gắng cắt văn bản thành các khối lớn dựa trên dấu ngắt đoạn (\n\n). Nếu một khối tạo ra vẫn vượt quá giới hạn chunk_size cho phép, nó sẽ đệ quy dùng dấu phân cách ở cấp độ tiếp theo (như ngắt dòng \n hoặc dấu chấm . ) để tiếp tục chia nhỏ. Cách tiếp cận "từ trên xuống" này giúp gom nhóm các câu có liên quan chặt chẽ với nhau thay vì cắt mù quáng theo số lượng ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu về xe điện VinFast (hướng dẫn sử dụng, thông số kỹ thuật, chính sách pin) có tính phân cấp cấu trúc rất cao, thường xuyên sử dụng các tiêu đề, gạch đầu dòng và các đoạn văn ngắn. RecursiveChunker khai thác triệt để pattern này bằng cách ưu tiên cắt ở các khoảng ngắt dòng, giúp bảo toàn trọn vẹn ngữ cảnh của một tính năng xe vào trong cùng một chunk. Điều này ngăn chặn việc thông số kỹ thuật bị chia cắt đứt đoạn, giúp mô hình AI trả lời chính xác và không bị "ảo giác" (hallucinate).


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Dữ liệu chính sách VinFast | RecursiveChunker <br> Size: 200 | 171 | 131.3 | Khá: Giữ được cấu trúc câu nhưng các chunk bị vỡ quá nhỏ (vụn). Khi hỏi về một tính năng phức tạp, AI phải nhặt từ quá nhiều chunk khác nhau dẫn đến câu trả lời thiếu ý. |
| Dữ liệu chính sách VinFast | Custom RecursiveChunker <br> Size: 500 | 58 | 415.5 | Chọn size=500 vì đây là mức vừa vặn để gói gọn trọn vẹn 1 ý tưởng giải thích tính năng xe VinFast. Khả năng cắt theo dấu ngắt đoạn (\n\n) của nó đã đủ tốt để bảo toàn ngữ cảnh. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 8/10 | Lấy trọn vẹn ngữ cảnh của từng tính năng. Vượt qua 80% bài test thực tế, cung cấp dữ liệu cực chuẩn cho AI mà không bị lẫn lộn thông số. | Sẽ hơi dư thừa text một chút nếu người dùng chỉ hỏi một câu rất ngắn gọn (như chỉ hỏi về giá). |
| Trang | FixedSizeChunker | 8/10 |- Giữ được context giữa các chunk nhờ overlap <br> - Cải thiện độ chính xác retrieval so với không overlap | - Tăng số lượng chunk → tốn tài nguyên hơn <br> - Có thể lặp lại thông tin |
| Phương | RecursiveChunker | 6/10 | | |
| Nghĩa | KeywordChunk | 6/10 | Kết quả tốt nếu keyword đúng | Phụ thuộc vào keyword |


**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi sử dụng Regex với mẫu r'([.!?]\s+|\.\n)' để nhận diện và tách câu, đồng thời vẫn giữ lại được các dấu câu nguyên bản (chấm, than, hỏi). Edge case được xử lý kỹ lưỡng ở đây là các khoảng trắng thừa ở hai đầu câu (dùng .strip()) và việc gom chính xác số lượng câu dựa trên max_sentences_per_chunk, đồng thời đảm bảo đoạn text lẻ ở cuối cùng không bị bỏ sót.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy hoạt động theo phương pháp "Top-Down", duyệt qua mảng separators theo thứ tự ưu tiên (từ ngắt đoạn \n\n đến khoảng trắng). Base case (điều kiện dừng) là khi độ dài của đoạn văn bản hiện tại nhỏ hơn hoặc bằng chunk_size, hàm sẽ trả về chính đoạn đó. Nếu lớn hơn, nó sẽ cắt bằng separator ưu tiên cao nhất tìm được và tiếp tục gọi đệ quy _split trên các mảnh vỡ.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Danh sách các tài liệu và vector được lưu trữ trên bộ nhớ (In-memory) dưới dạng các Dictionary/Danh sách đối tượng (chứa ID, text, metadata và embedding vector). Khi thực hiện search, hệ thống quét qua không gian lưu trữ và gọi hàm compute_similarity (Cosine Similarity) giữa vector của câu hỏi và từng vector tài liệu, sau đó sắp xếp giảm dần để trả về Top-K kết quả liên quan nhất.

**`search_with_filter` + `delete_document`** — approach:
> Tôi sử dụng cơ chế Pre-filtering (lọc metadata trước khi tính toán vector). Khi có yêu cầu, hệ thống sẽ rà soát và loại bỏ các tài liệu không khớp metadata, sau đó mới tính Cosine Similarity trên tập kết quả thu gọn để tối ưu hiệu năng. Hàm delete_document hoạt động bằng cách dò tìm doc_id tương ứng và xóa trực tiếp bản ghi đó khỏi cấu trúc dữ liệu lưu trữ (list/dict).

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc Prompt được thiết kế cực kỳ chặt chẽ với các chỉ thị rõ ràng (ví dụ. Context được inject (bơm) vào bằng cách nối chuỗi nội dung của các chunk trả về từ EmbeddingStore vào thẳng phần thân của prompt, kết hợp với các rào cản ngăn ngừa ảo giác nếu ngữ cảnh không chứa câu trả lời.

### Test Results

```
===================================================================== short test summary info =====================================================================
FAILED tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string - AssertionError: ('Answer based on context.', [{'id': 'd3', 'content': 'Vector databases store embeddings for similarity search.', 'metadata': {}, 'embedding': ...
FAILED tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive - KeyError: 'count'
FAILED tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length - AssertionError: 'count' not found in {'num_chunks': 8, 'avg_length': 99.38, 'min_length': 95, 'max_length': 100, 'chunks': ['Artificial intelligence is transfo...
FAILED tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies - AssertionError: 'fixed_size' not found in {'Naive_FixedSize': {'num_chunks': 8, 'avg_length': 99.38, 'min_length': 95, 'max_length': 100, 'chunks': ['Artificia...
FAILED tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size - AssertionError: 2 not less than 2
FAILED tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc - AssertionError: False is not true
================================================================== 6 failed, 36 passed in 0.91s ===================================================================
```

**Số tests pass:** 36 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sạc pin xe VF8 mất bao lâu thì đầy? | Thời gian nạp tối đa năng lượng cho VinFast VF 8 là mấy tiếng? | high | 0.9100 | Có |
| 2 | Tôi muốn mua xe điện VinFast VF8. | Tôi muốn bán xe điện VinFast VF8. | low | 0.9009 | Không |
| 3 | Hệ thống phanh tự động ADAS hoạt động rất mượt. | Chính sách thuê pin 10 năm của hãng. | low | 0.5785 | Có |
| 4 | Giá thay pin xe VF5 là bao nhiêu? | Xe VF5 có sạc pin tại nhà được không? | low | 0.8045 | Không |
| 5 | Trợ lý ảo Vivi không phản hồi lệnh của tôi. | Hệ thống nhận diện giọng nói trên xe đang bị lỗi. | high | 0.7168 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Pair 2 (Mua vs Bán) với điểm số cao chót vót 0.9009, và Pair 4 (Giá pin vs Sạc pin) lên tới 0.8045 dù chúng mang ý nghĩa hoàn toàn khác biệt.
>
> Điều này cho thấy AI không hiểu logic trái nghĩa hay ý định như con người, mà nó đánh giá dựa trên "ngữ cảnh xuất hiện". Vì các từ "mua/bán" hoặc "giá/sạc" thường xuyên xuất hiện cùng nhau trong các văn bản nói về "chính sách xe điện VinFast", vector của chúng bị xếp rất gần nhau trong không gian đa chiều. Điều này cho thấy hệ thống RAG thực tế không thể chỉ dựa 100% vào Vector Search mà nên kết hợp thêm tìm kiếm từ khóa để đảm bảo độ chính xác tuyệt đối.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chương trình thu xăng đổi điện kết thúc vào ngày nào? | 30/04/2026 |
| 2 | Tôi muốn mua lại pin thuê của xe VF e34 loại Gotion, giá năm 2025 là bao nhiêu? | 90.000.000 VNĐ |
| 3 | Tôi mua xe VF 8 ngày 15/02/2026 thì được ưu đãi sạc pin như thế nào? | 10 lần/tháng |
| 4 | VinFast có hỗ trợ đổi xe máy điện cũ sang ô tô điện VF 3 không? | Không, chỉ xe xăng được đổi |
| 5 | Ưu đãi giảm giá cho xe VF 9 sản xuất năm 2023 là bao nhiêu? | 250.000.000 VNĐ |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Chương trình thu xăng đổi điện kết thúc vào ngày nào? | gas_to_ev_0326.md | 0.7141 | No | ...đổi từ xe xăng sang xe điện VinFast kết thúc vào ngày 31/03/2026... |
| 2 | Tôi muốn mua lại pin thuê của xe VF e34 loại Gotion, giá năm 2025 là bao nhiêu? | battery_0326.md | 0.7731 | Yes | ...giá bán pin thuê của xe VF e34 loại Pin Gotion/SDI vào năm 2025 là **90.000.000 VNĐ**. |
| 3 | Tôi mua xe VF 8 ngày 15/02/2026 thì được ưu đãi sạc pin như thế nào? | charging_ev_0326.md | 0.8182 | Yes | ...ưu đãi miễn phí 10 lần sạc đầu tiên/xe/tháng... |
| 4 | VinFast có hỗ trợ đổi xe máy điện cũ sang ô tô điện VF 3 không? | gas_to_ev_0326.md | 0.7992 | Yes | ...không có thông tin về chính sách hỗ trợ đổi xe máy điện cũ sang ô tô điện VF3. |
| 5 | Ưu đãi giảm giá cho xe VF 9 sản xuất năm 2023 là bao nhiêu? | discontinued_models_0326.md | 0.7891 | Yes | Ưu đãi giảm giá cho xe VF 9 Plus sản xuất năm 2023 là 250.000.000 VNĐ. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được kỹ thuật viết biểu thức chính quy (Regex) cực kỳ khéo léo từ bạn cùng nhóm khi xử lý SentenceChunker. Việc bắt chính xác các trường hợp ngoại lệ trong tiếng Việt (như các từ viết tắt "TP.HCM", "VND" có chứa dấu chấm) giúp thuật toán không bị lầm tưởng đó là kết thúc câu, từ đó tránh việc các chunk bị cắt vụn một cách vô lý.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua phần demo, tôi ấn tượng với cách nhóm bạn áp dụng các độ đo định lượng như Precision, Recall và Accuracy để đánh giá chất lượng Retrieval thay vì chỉ test thủ công bằng mắt. Việc theo dõi chỉ số Recall giúp họ phát hiện ra những tài liệu bị hệ thống bỏ sót, trong khi Precision giúp tối ưu hóa kết quả trả về để tránh 'nhồi nhét' ngữ cảnh rác vào Prompt. Cách tiếp cận này giúp họ đưa ra các quyết định tinh chỉnh tham số một cách vô cùng thuyết phục.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu được làm lại, tôi sẽ nâng cấp thuật toán RecursiveChunker bằng cách bổ sung thêm cơ chế overlap (giữ lại khoảng 50 ký tự nối giữa 2 chunk) để giải quyết triệt để vấn đề đại từ thay thế bị đứt gãy.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 8 / 10 |
| Similarity predictions | Cá nhân | 3 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 25 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **84 / 100** |
