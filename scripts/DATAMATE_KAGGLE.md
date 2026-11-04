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

## Khắc phục OOM khi xử lý prompt dài

Model 4-bit vừa VRAM không có nghĩa mọi phép tính lúc inference đều vừa.
SDPA có thể dùng math backend với tensor trung gian lớn; chỉ đổi allocator
không giải quyết được một lần cấp phát lớn hơn VRAM còn trống.
Log OOM chưa có stack trace không đủ để khẳng định chính xác phép tính gây lỗi.

Backend Qwen dùng `prefill_chunk_size=256`, KV cache động và `logits_to_keep=1`.
Prompt được xử lý từng khối, giữ nguyên nội dung và giới hạn 12.288 token.
Cách này giảm kích thước tensor trung gian lúc đọc prompt, có thể chậm hơn;
không bảo đảm mọi cấu hình đều hết OOM. Tham số được ghi trong thông tin model.
Gemma không áp dụng các tùy chọn Qwen này.

Sau khi cập nhật từ notebook cũ:
1. Tải kết quả cũ nếu cần, dừng và khởi động lại session Kaggle để bỏ model cũ.
2. Import notebook mới, bật GPU và Internet, gắn dataset rồi Run All.
3. Dùng thư mục đầu ra mới, `RESUME=False`: mã/cấu hình khác lượt cũ nên không resume.
4. Nếu vẫn OOM, khởi động lại session, đặt `PREFILL_CHUNK_SIZE=128`, chạy vào
   thư mục mới. Lưu `trace.jsonl` có traceback đầy đủ để xác định nơi cấp phát lỗi.

Không cần giảm `MAX_INPUT_TOKENS` trước: rulebook dài có thể gây `ContextOverflow`.
Không chỉ chạy lại ô nạp model khi model cũ còn trên GPU. GPU thứ hai chưa được dùng.
Chưa xác thực GPU Kaggle trong môi trường phát triển; kiểm thử CPU kiểm tra
truyền tham số và giữ nguyên prompt, không chứng minh mức dùng VRAM thực tế.
Tham khảo [chunked prefill trong Transformers 5.5.0](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/generation/utils.py)
và [SDPA của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

Notebook tải script và model qua mạng, **không phải Docker offline nộp cuộc thi**.
Không dùng test để huấn luyện.
Đối chiếu nhãn chỉ là tham khảo và chỉ thực hiện sau khi đã tạo dự đoán.

Chạy kiểm thử CPU: `python -m unittest discover -s tests -v`.
