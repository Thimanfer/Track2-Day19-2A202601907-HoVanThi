# Báo Cáo Thu Hoạch (Reflection) — Lab 19

**Họ và tên:** Hồ Văn Thi  
**Khóa / Cohort:** A20 - K3 (Track 2)  
**Môi trường thực thi (Path):** Lite path (fastembed + Qdrant in-memory + Feast SQLite + FastAPI)

---

## 1. Trả Lời Câu Hỏi Thu Hoạch (≤ 200 chữ)

Sau khi chạy benchmark trên 50 golden queries, em thấy mỗi mode có ưu điểm khác nhau tùy vào dạng câu hỏi.

* **`exact`:** BM25 và Hybrid cho kết quả tốt nhất, cùng đạt 96.7%. Với dạng query này, các từ khóa kỹ thuật thường xuất hiện đúng như trong dữ liệu nên BM25 hoạt động khá hiệu quả. Vector thuần có điểm thấp hơn vì không tận dụng được việc trùng từ khóa.
* **`paraphrase`:** Kết quả của cả hai phương pháp đều giảm. Trong quá trình chạy, em nhận thấy model `bge-small` đang dùng phù hợp với tiếng Anh hơn nên khả năng hiểu các cách diễn đạt khác nhau chưa thực sự tốt. Tuy nhiên, Vector vẫn có lợi thế hơn ở việc tìm các câu có ý nghĩa gần nhau.
* **`mixed`:** Hybrid cho kết quả tốt nhất với 100.0%, cao hơn BM25 (97.0%) và Vector (98.5%). Điều này có thể giải thích là Hybrid kết hợp được cả matching theo từ khóa và matching theo ngữ nghĩa.

Theo em, không phải lúc nào cũng cần dùng Hybrid. Với việc tìm mã lỗi, mã SKU hoặc tên hàm/API thì **BM25** có thể phù hợp hơn vì yêu cầu chính xác về từ khóa. Ngược lại, nếu câu hỏi mang tính ngữ nghĩa nhiều và không có từ khóa trùng trực tiếp thì **Vector** sẽ có lợi thế hơn.

---

## 2. Điều Khiến Em Bất Ngờ Nhất Khi Làm Lab Này

Điều em thấy đáng chú ý nhất là vấn đề **data leakage** ở NB8. Ban đầu nhìn kết quả AUC khi train offline thì khá tốt nên em nghĩ model đang hoạt động ổn. Nhưng khi xem kỹ cách join dữ liệu, em nhận ra việc dùng Latest join thay cho Point-in-Time (PIT) join có thể làm dữ liệu tương lai bị đưa vào quá trình train.

Điều này khiến kết quả đánh giá cao hơn thực tế. Nếu đem model đó lên production thì khả năng cao kết quả sẽ không còn tốt như lúc test. Qua phần này, em hiểu rõ hơn tại sao khi làm machine learning không chỉ cần quan tâm đến model mà còn phải kiểm tra kỹ cách chuẩn bị và lấy dữ liệu.

---

## 3. Bonus Challenge

* [x] Đã hoàn thành phần Bonus Challenge (chi tiết nằm trong thư mục `bonus/`)
* [ ] Pair work với: *Em làm độc lập*
