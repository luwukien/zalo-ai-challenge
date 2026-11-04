# Chạy DataMate với Qwen trên Kaggle

Notebook: `notebooks/datamate_qwen_kaggle.ipynb`. Thử nghiệm dùng
`Qwen/Qwen3-4B-Instruct-2507` (4B, Apache-2.0), nạp 4-bit NF4 lên GPU 0.
Kaggle T4 x2 có hai GPU, nhưng notebook hiện chỉ dùng GPU đầu tiên.

1. Đưa thư mục dữ liệu đã chia từ Drive lên **Kaggle Dataset riêng tư**.
   Giữ cấu trúc `ten-thu-muc/train/questions.jsonl`, `train/rulebook.md`,
   `train/calendar.csv`, `train/data/` và tương tự cho `validation/`.
   Có thể chỉ thêm `train/` khi thử 5 câu đầu. Không tải dữ liệu lên GitHub.
2. Tạo Kaggle Notebook, import file notebook này. Trong Settings bật Internet,
   chọn Accelerator **GPU T4 x2**, rồi bấm **Add Input** để gắn dataset.
3. Chạy các ô theo thứ tự. Mặc định dùng 5 câu train. Notebook tìm đường dẫn
   `train/` trong `/kaggle/input`; nếu có nhiều dataset phù hợp, điền `DATA_ROOT`
   trong ô cấu hình thành thư mục cha của `train/`.
4. Kết quả được ghi vào `/kaggle/working/datamate_runs/`: `predictions.jsonl`,
   `trace.jsonl`, `summary.json`, code theo từng câu và các checkpoint. Tải kết quả
   từ phần Output của Kaggle; phiên làm việc có thể bị xóa khi kết thúc.
5. Muốn thử validation, đổi `SPLIT` thành `validation` và `LIMIT` thành số câu
   cần thử hoặc `None`, chọn thư mục đầu ra mới rồi chạy lại.

Khi nạp model hoặc sinh code bị OOM, notebook sẽ báo lỗi; cấu hình 4-bit và
T4 chưa được xác thực GPU trong môi trường phát triển này. Chụp nguyên lỗi để
điều chỉnh tiếp. Lần thử có tải script và model qua mạng: đây là notebook phát
triển, **không phải Docker offline nộp cuộc thi**. Không dùng test để huấn luyện.
Đối chiếu nhãn chỉ là tham khảo và chỉ thực hiện sau khi đã tạo dự đoán.

Chạy kiểm thử CPU: `python -m unittest discover -s tests -v`.
