# DataMate Qwen trên Kaggle: baseline 6 bước

Notebook: `notebooks/datamate_qwen_kaggle.ipynb`.
Dùng `Qwen/Qwen3-4B-Instruct-2507`, NF4 4-bit trên GPU 0.

## Cách chạy

1. Import notebook vào Kaggle, bật Internet và GPU T4 x2.
2. Add Input dataset riêng tư chứa `train/` hoặc `validation/` với
   `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/README.md`
   và các file nghiệp vụ. `answers.jsonl` là tùy chọn, chỉ dùng đánh giá cuối.
3. Mặc định `QUESTION_INDEX=0`, `RUN_BATCH=False`: chạy các cell theo thứ tự
   để quan sát một câu. `DATA_ROOT=None` tự tìm thư mục cha của split;
   nếu tìm thấy nhiều dataset thì điền đường dẫn cụ thể.
4. Sau khi kiểm tra một câu, bật `RUN_BATCH=True`, đặt `LIMIT=5` hoặc `None`
   để chạy toàn bộ split. Batch chạy độc lập, có thể chạy lại câu debug.
5. Validation: đổi `SPLIT="validation"`, chọn giới hạn và chạy phiên mới.

## Sáu bước và bằng chứng quan sát

| Bước | Công việc | Đầu ra |
| --- | --- | --- |
| 1. Explore Environment | Python kiểm tra thư viện, GPU, VRAM | Thông tin môi trường |
| 2. Load Data Structure | Đọc tài liệu, nạp bảng không sửa dữ liệu gốc | Schema, số dòng, mẫu, nguồn |
| 3. Assimilate Domain Knowledge | Agent đọc nghĩa dữ liệu, luật và điểm chưa rõ | JSON có trích dẫn; Python probe tùy chọn |
| 4. Extract Insights | Agent sinh/chạy Python trên toàn bộ dữ liệu để kiểm tra luật | Thống kê và quyết định ready/reread/explore/unanswerable |
| 5. Formulate Plan | Chọn bảng/cột và thứ tự thao tác | JSON kế hoạch và điều kiện kiểm tra |
| 6. Execute | Sinh/chạy Python theo kế hoạch và sửa lỗi có giới hạn | Dự đoán, stdout, traceback |

Bước 4 có thể quay lại bước 3 hoặc chạy probe mới; tối đa `MAX_INSIGHT_ROUNDS=3`.
Python không bị giới hạn ở bước 6: bước 1/2 dùng Python, bước 3 cho phép agent
sinh probe, bước 4 yêu cầu agent chạy probe trước khi quyết định đủ bằng chứng.
Công cụ `run_python` dùng chung cho các bước. Mỗi lời gọi model có tối đa
`MAX_REPAIRS=2` lượt sửa sau lần đầu. Hết lượt khám phá/sửa là lỗi kỹ thuật
`abstain`; `unanswerable` dành cho thiếu dữ liệu nghiệp vụ có bằng chứng.
Kiểm tra schema và trích dẫn không chứng minh suy luận nghiệp vụ đúng.

Notebook cũ quan sát được 5/5 câu lỗi selection. Trace câu đầu trả JSON
đáp án thay vì JSON chọn bảng/cột cả bốn lần. Notebook mới tách hợp đồng
mỗi bước vào system prompt và cuối user prompt, phân biệt tài liệu nghiệp vụ
với chỉ thị định dạng. Không ép JSON đáp án thành JSON kế hoạch.

## Debug và kết quả

Logic điều phối, prompt và vòng lặp nằm ngay trong notebook.
Loader, helper tiền/ngày, worker và Qwen backend được tải từ commit cố định
`4b8f456b2c68da8093439c25a4cea495ef124325` của repository.
Không gọi pipeline hai bước `run_baseline` cũ.

Kết quả nằm dưới `/kaggle/working/datamate_runs/<run>/`:

- `manifest.json`: cấu hình, phiên bản thư viện, thông tin model và helper commit.
- `<question_id>/<attempt_time>/trace.json`: prompt, raw response, token, thời gian,
  code, kết quả thực thi, stdout/stderr và traceback.
- Cùng thư mục trace: `readme.md`, `rulebook.md`, các file `code_*.py`.
- `predictions.jsonl`: dự đoán của lượt debug/batch được xuất tại cell báo cáo.
- `evaluation.json`: đối chiếu tham khảo, không phải bộ chấm chính thức.

Trong chế độ từng cell, lỗi dừng ở cell tương ứng để kiểm tra trace.
Batch chuyển lỗi từng câu thành abstain; lỗi GPU/context dừng cả batch.
Câu đang dừng vẫn có trace, không được xuất thành dự đoán hoàn thành.
Đáp án tham khảo chỉ đọc sau khi ghi predictions, không được đưa vào prompt.

Chạy lại bước trước sẽ vô hiệu hóa kết quả phụ thuộc phía sau. Muốn đổi câu,
đổi `QUESTION_INDEX`, chạy lại cell `load_question` rồi các bước 3–6.
Bản này không tự resume: mỗi lượt có thư mục riêng. Tải Output về trước khi
kết thúc phiên Kaggle. Cell timeline cuối đọc lại trace mà không gọi model.

## Giới hạn và kiểm thử

Giữ chunked prefill 256 token, dynamic KV cache và logits token cuối từ backend cũ.
Không cắt âm thầm luật khi vượt 12.288 input token: dừng bằng ContextOverflow.
OOM: restart session, thử `PREFILL_CHUNK_SIZE=128` và thư mục mới; không nạp model
chồng lên model cũ. Chỉ GPU đầu tiên được dùng. Chưa xác thực GPU Kaggle ở môi trường
phát triển; chưa có số đo accuracy hay VRAM cho pipeline mới.

Worker dùng subprocess, AST guard và timeout; không phải sandbox hoàn chỉnh.
Đây là notebook thử nghiệm có tải mã/model qua Internet, chưa phải Docker offline.
Không commit dữ liệu, weights hay log. Notebook lưu trong Git phải sạch output.

Chạy `python -m unittest discover -s tests -v` từ thư mục dự án.
`tests/test_datamate_notebook.py` chạy trực tiếp các hàm trong cell bằng model giả
và dữ liệu tổng hợp: thành công, sai schema, đọc lại luật, probe trước Execute,
sửa Python, hết vòng khám phá và dừng GPU/context; không tải model hay đọc dataset thật.
