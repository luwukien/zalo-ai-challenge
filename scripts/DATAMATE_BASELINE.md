# DataMate baseline: Gemma 4 E4B trên Colab

[Mở notebook trên Colab](https://colab.research.google.com/github/luwukien/zalo-ai-challenge/blob/codex/datamate-baseline/notebooks/datamate_baseline_colab.ipynb).

Baseline dùng hai bước prompt: chọn bảng/cột và kiểu đáp án, rồi sinh hàm Python để
tính trên toàn bộ dữ liệu của các bảng được chọn. Lỗi chọn schema, cú pháp, thực thi
và định dạng đầu ra được sửa trong ngân sách chung tối đa 3 lần mỗi câu.
Không fine-tune, không dùng vector database và không gửi dữ liệu tới API LLM bên ngoài.

## Chạy thử 5 câu

1. Trong Colab, chọn **Runtime → Change runtime type → GPU**. Mục tiêu là T4 16 GB;
   chương trình yêu cầu ít nhất 10 GiB VRAM trống trước khi thử nạp model. Đây là
   kiểm tra ban đầu, không bảo đảm mọi prompt đều vừa bộ nhớ.
2. Chuẩn bị cấu trúc Drive như dưới đây, hoặc sửa `DATA_ROOT` trỏ tới split đã có.
3. Giữ `SPLIT='train'`, `LIMIT=5`, rồi chạy các ô cài đặt, mount Drive, nạp model,
   inference và xem kết quả. Các câu được lấy theo thứ tự trong `questions.jsonl`.
4. Nếu Colab yêu cầu khởi động lại runtime sau cài đặt, làm theo rồi chạy lại các ô.
   Token Hugging Face là tùy chọn: đặt trong Colab Secrets tên `HF_TOKEN` nếu cần,
   không chép token vào notebook.

```text
/content/drive/MyDrive/datasets/split_colab_seed_42/
  train/
    questions.jsonl
    answers.jsonl         # chỉ đối chiếu sau dự đoán; có thể không có
    rulebook.md
    calendar.csv
    data/biz_.../README.md
    data/biz_.../<CSV, XLSX, SQLite>
  validation/
    <cấu trúc tương tự>
```

Notebook tải mã và requirements từ cùng một commit đã resolve của nhánh
`codex/datamate-baseline`; `CODE_REF` có thể là commit SHA để cố định phiên bản.
Chỉ tải mã, không tải dataset từ GitHub. Dữ liệu, cache, weights và logs không được
commit vào repository.

Model cố định mặc định là `google/gemma-4-E4B-it`, nạp qua `AutoProcessor` và
`AutoModelForMultimodalLM` với bitsandbytes NF4, double quantization, SDPA,
FP16 trên T4 hoặc BF16 khi GPU hỗ trợ. Không có fallback CPU/model khác.
Giữ bản PyTorch CUDA của Colab; ô cài đặt không chủ động nâng cấp PyTorch.
Phiên bản thư viện và revision thực tế của model được ghi trong manifest/summary.

## Validation và tiếp tục lượt chạy

Ô cuối notebook mặc định có `RUN_VALIDATION=False`. Đổi thành `True`, chọn
`VALIDATION_LIMIT=20` hoặc `None` để chạy toàn bộ. Ô này dùng lại model trên GPU,
đọc rulebook/calendar/README từ validation và tạo thư mục kết quả khác train.

Sau mất kết nối, đặt `OUTPUT_DIR` về thư mục cũ và `RESUME=True`. Giữ nguyên
code, model revision, phiên bản thư viện, split, seed và tham số inference.
Có thể tăng `LIMIT`, nhưng không giảm xuống dưới các câu đã thử. SHA-256 kiểm tra
toàn bộ đầu vào inference của split; labels không tham gia fingerprint.
Câu hoàn thành (kể cả abstain do hết lượt sửa) được giữ nguyên. Câu đang dở chạy lại.
Trace giữ lượt bị ngắt trong `previous_runs`; file code cũ không bị ghi đè.
Muốn thử lại câu đã hoàn thành hoặc đổi cấu hình, tạo thư mục đầu ra mới.

Giới hạn mặc định: input 12.288 token, output 2.048 token mỗi lần sinh,
timeout Python 60 giây (gồm khởi động worker và deserialize), seed 42.
Lần đầu greedy; sửa lỗi sampling `temperature=1.0`, `top_p=0.9`.
Thinking mặc định tắt. Seed ổn định theo câu, giai đoạn và lần thử;
không cam kết tái lập bit-for-bit trên GPU/thư viện khác nhau.

Prompt giảm dòng mẫu và few-shot khi quá dài; không cắt câu hỏi, README hay rulebook.
Nếu vẫn quá giới hạn, câu có lỗi `ContextOverflow` và `abstain`. Có thể tăng giới hạn
trên GPU đủ bộ nhớ và tạo lượt chạy mới. Lỗi CUDA/OOM dừng cả lượt và giữ tiến độ;
không đưa lỗi GPU vào vòng sửa Python. Để giảm bộ nhớ, hạ giới hạn input/output hoặc
dùng GPU lớn hơn, giữ nguyên Gemma E4B. Thay cấu hình cần thư mục mới.

## Dữ liệu và hợp đồng hàm

- Đọc README trước dữ liệu. CSV dùng UTF-8/UTF-8 BOM và giữ chuỗi, bao gồm mã có số 0 đầu.
- XLSX đọc tất cả sheet với `data_only=True`; ô công thức cần có giá trị cache trong file.
  Tìm header trong 10 dòng đầu, ưu tiên tên cột README; nếu không có mô tả cột,
  chọn dòng chữ đầy đủ, duy nhất và rộng nhất, ưu tiên dòng xuất hiện trước.
- Ghép các sheet tháng và file đánh số/năm thuộc cùng họ được README mô tả, chỉ khi
  schema khớp. Sheet khác chức năng giữ riêng. Schema không khớp báo lỗi thay vì bỏ qua.
- SQLite mở `mode=ro` và `query_only=ON`; mỗi bảng thành DataFrame.
- Bảng có khóa ổn định như `SoBanHang`, `DuLieuXuat::DanhSachThuoc`, `pos::don_hang`.
  Ba cột nguồn là `__source_file`, `__source_sheet`, `__source_row`.
  SQLite dùng số thứ tự hàng đọc được, không coi đó là khóa nghiệp vụ.
- Giữ toàn bộ dòng, tên cột và giá trị gốc. Không tự forward-fill, loại trùng, đổi
  tiền/ngày hay lọc theo `as_of_date` ở loader. Code sinh ra phải áp dụng rulebook.
  Cache tối đa hai doanh nghiệp trong RAM để giới hạn bộ nhớ phiên Colab.

```python
def answer(tables, as_of_date, calendar):
    # tables: dict[str, pandas.DataFrame], đủ mọi dòng của bảng được chọn
    # as_of_date: YYYY-MM-DD; calendar: DataFrame lịch âm/dương dùng chung
    return {"type": "number", "answer": 123}
```

Các helper có sẵn: `parse_money`, `parse_number`, `parse_date`, `round_half_up`.
Tiền hỗ trợ `55.000đ`, `38,000`, `25k`, `18,914tr`; có hậu tố k/tr thì phần trước
hậu tố là số thập phân, không ngầm coi ba chữ số cuối là phân cách nghìn.
Không hậu tố, một dấu và ba chữ số sau dấu được coi là phân cách nghìn;
truyền `decimal_separator='.'` hoặc `','` nếu cần cách hiểu khác.
Giá trị rỗng thành NaN/NaT, chuỗi không đọc được gây lỗi, không đổi thành 0.
`parse_money` không làm tròn từng dòng; code phải làm tròn kết quả cuối theo rulebook.
`parse_date` nhận ngày Excel, ISO, dd/mm/yyyy; dd/mm cần truyền `year` hoặc `sheet`.

Đầu ra kiểm tra nghiêm kiểu number, percent, boolean, string, date, list, table,
unanswerable và abstain. Loại bỏ NaN/Infinity, không biến chuỗi số thành JSON number.
Model trả kiểu theo schema DataMate; runner thêm `id` từ câu hỏi.
Lỗi kỹ thuật cuối cùng thành abstain; unanswerable dành cho thiếu dữ liệu nghiệp vụ.

## Logs và đối chiếu

Thư mục mặc định: `/content/drive/MyDrive/datamate_runs/<run_name>/`.

| File | Nội dung |
| --- | --- |
| `manifest.json` | Cấu hình inference, hash dữ liệu/mã, model revision và môi trường |
| `questions/<id>.json` | Checkpoint từng câu, ghi sau mỗi bước/lần thử |
| `predictions.jsonl` | Mỗi câu hoàn thành một dòng `id`, `type`, `answer` nếu có |
| `trace.jsonl` | Câu hỏi, selection, phản hồi model, code, traceback, token, thời gian |
| `code/<id>/attempt_<n>.py` | Python đã sinh, gồm cả bản lỗi |
| `summary.json` | Trạng thái lượt chạy, thời gian nạp model/phiên, thống kê, đối chiếu |

Thời gian mỗi câu tách đọc dữ liệu, sinh model, thực thi và tổng; tổng còn gồm
xây prompt, đếm token và lưu checkpoint. Thời gian nạp model ghi riêng.
File checkpoint từng câu là nguồn khôi phục; hai file JSONL được dựng lại sau mỗi câu.

`answers.jsonl` được đọc **sau khi đã ghi predictions**, không dùng để chọn schema,
sinh prompt, sửa code hay quyết định khi nào dừng sửa. Đối chiếu tham khảo xét type,
số với sai số tuyệt đối 1e-6, chuỗi chuẩn hóa Unicode/hoa thường/khoảng trắng/dấu câu,
alias ghi trong labels, list đúng thứ tự và bảng không phụ thuộc thứ tự hàng.
Đây không phải bộ chấm chính thức: chưa suy ra mọi alias từ dữ liệu doanh nghiệp;
cột bảng đối chiếu theo vị trí. Không diễn giải tỷ lệ này thành điểm leaderboard.

Code được chạy trong subprocess CPU với bộ lọc AST, danh sách import hạn chế,
builtins hạn chế, không chuyển token Hugging Face qua environment và có timeout.
Các biện pháp này giúp chặn lỗi I/O phổ biến, **không phải sandbox bảo mật hoàn chỉnh**.
Pandas/NumPy là thư viện Python mạnh; không dùng runtime này để chạy mã độc hoặc dữ liệu
không tin cậy. Code chỉ nhận bảng đã nạp, không cần đường dẫn tới Drive.

## Kiểm thử và CLI

Kiểm thử CPU chỉ cần pandas, numpy, openpyxl; không tải model:

```powershell
python -m unittest discover -s tests -v
python scripts/datamate_baseline.py --data-root datasets/split --split train --limit 5 --inspect-only
python scripts/datamate_baseline.py --data-root datasets/split --split validation --limit 0 --inspect-only
```

`--inspect-only` đọc dữ liệu để báo số bảng/dòng, không ghi kết quả và không dùng GPU.
Trên máy Linux có GPU và các requirements:

```bash
python scripts/datamate_baseline.py --data-root /path/to/split --split train --limit 5 --output-dir outputs/train_5
python scripts/datamate_baseline.py --data-root /path/to/split --split validation --limit 20 --output-dir outputs/validation_20
```

Test dùng dữ liệu giả kiểm tra loader, schema, phép nối, sửa lỗi, timeout, chống đọc gold
trước dự đoán và resume. Máy phát triển hiện không có GPU; các kiểm tra CPU không
chứng minh Gemma nạp được hoặc đạt accuracy trên Colab. Nghiệm thu GPU là chạy 5 câu
train bằng notebook và kiểm tra đầy đủ code/lỗi/timing; chưa đặt ngưỡng accuracy.

## Nguồn và khác biệt so với paper

- [AILS-NTUA at SemEval-2025 Task 8](https://aclanthology.org/2025.semeval-1.188/):
  áp dụng Main Module và Error-Fixing Module, metadata cột và dòng mẫu.
- [Gemma E4B model card](https://huggingface.co/google/gemma-4-E4B-it): model ID,
  Transformers API, chat template và chế độ thinking.
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes): NF4.

DataMate có nhiều bảng và quy tắc riêng nên baseline bổ sung chọn bảng, loader hỗn hợp,
README/rulebook/calendar và schema đáp án DataMate. Dùng 3 ví dụ tổng hợp ngắn thay vì
9 exemplar của paper, model Gemma thay model trong paper; không tuyên bố tái lập kết quả paper.
