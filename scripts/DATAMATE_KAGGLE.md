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
| 3. Assimilate Domain Knowledge | Agent đọc nghĩa dữ liệu, luật và điểm chưa rõ | JSON có trích dẫn và điểm chưa rõ; không chạy code |
| 4. Extract Insights | Agent sinh/chạy Python trên toàn bộ dữ liệu để kiểm tra luật | Thống kê và quyết định ready/reread/explore/unanswerable |
| 5. Formulate Plan | Chọn bảng/cột và thứ tự thao tác | JSON kế hoạch và điều kiện kiểm tra |
| 6. Execute | Sinh/chạy Python theo kế hoạch và sửa lỗi có giới hạn | Dự đoán, stdout, traceback |

Bước 4 có thể quay lại bước 3 hoặc chạy probe mới; tối đa `MAX_INSIGHT_ROUNDS=3`.
Python không bị giới hạn ở bước 6: bước 1/2 dùng Python để khám phá môi trường/cấu trúc,
bước 3 chỉ đọc hiểu luật, bước 4 yêu cầu agent chạy probe trước khi quyết định đủ bằng chứng.
Công cụ `run_python` dùng chung cho các bước. Mỗi lời gọi model có tối đa
`MAX_REPAIRS=2` lượt sửa sau lần đầu. Hết lượt khám phá/sửa là lỗi kỹ thuật
`abstain`; `unanswerable` dành cho thiếu dữ liệu nghiệp vụ có bằng chứng.
Kiểm tra schema và trích dẫn không chứng minh suy luận nghiệp vụ đúng.

Notebook cũ quan sát được 5/5 câu lỗi selection. Trace câu đầu trả JSON
đáp án thay vì JSON chọn bảng/cột cả bốn lần. Notebook mới tách hợp đồng
mỗi bước vào system prompt, cuối user chỉ nhắc tuân thủ, phân biệt tài liệu nghiệp vụ
với chỉ thị định dạng. Không ép JSON đáp án thành JSON kế hoạch.

## Debug và kết quả

Logic điều phối, prompt và vòng lặp nằm ngay trong notebook.
Loader, helper tiền/ngày, worker và Qwen backend được tải từ commit cố định
`1ebde592e4922cdc5fd3790e3031f8654f2fe57d` của repository.
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
Bước 3 dùng catalog gọn, không đưa dòng mẫu vào prompt. Bước khác thử catalog đầy đủ,
ngay khi vượt 12.288 input token thì bỏ mẫu và metadata nguồn lặp; giữ mọi bảng/cột,
dtype, số dòng, tên file/sheet. Hợp đồng chỉ gửi một lần trong system prompt.
Rulebook, README, câu hỏi, bằng chứng, kế hoạch và phản hồi sửa lỗi giữ nguyên.
Trace ghi `context_attempts` và `compact_catalog`; nếu vẫn vượt thì dừng ContextOverflow.
Không tự tăng ngân sách VRAM hay cắt luật. Khi cập nhật trong session đang chạy,
thay và chạy lại cell định nghĩa `BASE_SYSTEM`/`context_for`/`ask`, rồi bước 3;
không cần nạp lại model.
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

## Cập nhật trong session Kaggle đang chạy

Thêm một cell sau đây vào notebook đang dùng, chỉ cần thêm một lần:

```python
import urllib.request
sync_url = "https://raw.githubusercontent.com/luwukien/zalo-ai-challenge/codex/datamate-baseline/scripts/datamate_kaggle_sync.py"
sync_module = {}
with urllib.request.urlopen(sync_url, timeout=60) as response:
    exec(compile(response.read(), sync_url, "exec"), sync_module)
sync_module["sync_notebook"](globals())
```

Sau mỗi bản sửa đã push, chạy cell này khi kernel rảnh, rồi chạy lại từ bước 3.
Không cần import notebook hay tải lại model. Không chạy lại các cell định nghĩa cũ
ở phía trên: nội dung hiển thị trong giao diện không tự đổi, và chạy chúng sẽ ghi đè
hàm vừa đồng bộ. Nếu cần đọc code hiện hành, xem snapshot trong
`OUTPUT_DIR/code_updates/<commit>.ipynb`.

Script resolve branch thành commit SHA rồi tải notebook tại commit đó; chỉ thực thi
các cell có tag `definitions`. Không chạy cell config, pip, nạp model, batch hay chấm điểm.
Model, cấu hình và bảng đã nạp được giữ; kiến thức/insights/plan/prediction cũ bị vô hiệu hóa,
trace giữ lịch sử cùng commit cập nhật. Code không đổi thì không vô hiệu hóa state.
Mạng/tải code/biên dịch lỗi trước khi thay hàm sẽ giữ code cũ. Internet Kaggle phải bật.

Đây là thực thi code tin cậy từ repository dự án, không phải sandbox. Không có polling
nền: push GitHub không tự thay kernel đang mở. Thay đổi dependencies, helper backend,
model hoặc cấu hình không thuộc cơ chế này và vẫn cần cập nhật riêng. Không dùng
cơ chế cập nhật giữa lượt inference; nó chỉ chạy khi bạn bấm cell cập nhật.
