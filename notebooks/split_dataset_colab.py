# Copy toàn bộ file này vào một ô Google Colab rồi chạy.
# Colab sẽ yêu cầu bạn cấp quyền kết nối Google Drive.
# Mặc định: dữ liệu nguồn nằm trong /content/drive/MyDrive/datasets/train.
# Kết quả: /content/drive/MyDrive/datasets/split_colab_seed_42.
# Chia theo doanh nghiệp: 10% doanh nghiệp cho validation, làm tròn lên.
# Không ghi đè thư mục đầu ra đã tồn tại.
# Nếu questions.jsonl nằm trực tiếp trong datasets, sửa cấu hình thành:
# INPUT_DIR = DATASETS_DIR
# OUTPUT_DIR = DATASETS_DIR.parent / f'datasets_split_colab_seed_{SEED}'

# 1. Kết nối Google Drive
from google.colab import drive

drive.mount('/content/drive')

# 2. Cấu hình: sửa đường dẫn hoặc seed tại đây
from pathlib import Path

DATASETS_DIR = Path('/content/drive/MyDrive/datasets')
SEED = 42
INPUT_DIR = DATASETS_DIR / 'train'
OUTPUT_DIR = DATASETS_DIR / f'split_colab_seed_{SEED}'

print(f'Nguồn: {INPUT_DIR}')
print(f'Đầu ra: {OUTPUT_DIR}')
print(f'Seed: {SEED}')

# 3. Hàm chia dữ liệu
import json
import os
from pathlib import Path
import random
import re
import shutil

SHARED_FILES = ("calendar.csv", "rulebook.md")


def read_records(path, *, questions=False):
    """Index by ID, retaining input order and original JSON to avoid value changes."""
    records = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            fields = ("id", "business_id", "question", "as_of_date") if questions else ("id", "type")
            for field in fields:
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{location}: {field!r} must be a nonempty string")
            if questions:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", record["business_id"]):
                    raise ValueError(f"{location}: unsafe business_id {record['business_id']!r}")
            elif "answer" not in record:
                raise ValueError(f"{location}: missing 'answer' field")
            record_id = record["id"]
            if record_id in records:
                raise ValueError(f"{location}: duplicate ID {record_id!r}")
            records[record_id] = (record, line.rstrip("\r\n"))
    return records


def split_dataset(input_dir, output_dir, seed=42):
    """Validate all records and paths before creating the output directory."""
    input_dir = Path(input_dir).expanduser().resolve()
    requested_output = Path(output_dir).expanduser()
    output_dir = requested_output.resolve()
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        raise ValueError("Input and output directories must not overlap")
    if os.path.lexists(requested_output) or output_dir.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    questions = read_records(input_dir / "questions.jsonl", questions=True)
    answers = read_records(input_dir / "answers.jsonl")
    missing = questions.keys() - answers.keys()
    extra = answers.keys() - questions.keys()
    if missing or extra:
        raise ValueError(
            f"Question/answer IDs do not match: {len(missing)} missing answers, "
            f"{len(extra)} extra answers"
        )
    businesses = sorted({record["business_id"] for record, _ in questions.values()})
    if len(businesses) < 2:
        raise ValueError("At least two businesses are required")
    for filename in SHARED_FILES:
        if not (input_dir / filename).is_file():
            raise ValueError(f"Missing supporting file: {input_dir / filename}")
    for business in businesses:
        business_dir = input_dir / "data" / business
        if not business_dir.is_dir():
            raise ValueError(f"Missing business directory: {business_dir}")
        if not business_dir.resolve().is_relative_to(input_dir):
            raise ValueError(f"Business directory resolves outside input: {business_dir}")

    # Integer ceiling avoids float rounding; sorted IDs make sampling independent
    # of the order in which businesses first appear in questions.jsonl.
    validation_count = (len(businesses) + 9) // 10
    validation_businesses = set(random.Random(seed).sample(businesses, validation_count))
    assignments = {
        "train": [b for b in businesses if b not in validation_businesses],
        "validation": sorted(validation_businesses),
    }
    split_ids = {"train": [], "validation": []}
    for record_id, (record, _) in questions.items():
        subset = "validation" if record["business_id"] in validation_businesses else "train"
        split_ids[subset].append(record_id)

    manifest = {
        "strategy": "business_holdout",
        "seed": seed,
        "input_dir": str(input_dir),
        "validation_business_fraction": 0.1,
        "total_businesses": len(businesses),
        "total_pairs": len(questions),
        "actual_validation_pair_fraction": len(split_ids["validation"]) / len(questions),
        "splits": {
            subset: {
                "business_ids": assignments[subset],
                "business_count": len(assignments[subset]),
                "pair_count": len(split_ids[subset]),
            }
            for subset in assignments
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    for subset, business_ids in assignments.items():
        subset_dir = output_dir / subset
        (subset_dir / "data").mkdir(parents=True)
        for filename, records in (("questions.jsonl", questions), ("answers.jsonl", answers)):
            with (subset_dir / filename).open("w", encoding="utf-8", newline="\n") as target:
                for record_id in split_ids[subset]:
                    target.write(records[record_id][1] + "\n")
        for filename in SHARED_FILES:
            shutil.copy2(input_dir / filename, subset_dir / filename)
        for business in business_ids:
            shutil.copytree(input_dir / "data" / business, subset_dir / "data" / business)

    # Written last: a manifest indicates that all output files were copied.
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return manifest

# 4. Chia và lưu kết quả
print('Đang kiểm tra đầu vào và sao chép dữ liệu...', flush=True)
manifest = split_dataset(INPUT_DIR, OUTPUT_DIR, seed=SEED)

for subset, details in manifest['splits'].items():
    print(f"{subset}: {details['pair_count']} cặp, {details['business_count']} doanh nghiệp")
print(f"Tỷ lệ câu hỏi validation: {manifest['actual_validation_pair_fraction']:.2%}")
print('Doanh nghiệp validation:', ', '.join(manifest['splits']['validation']['business_ids']))
print('Manifest:', OUTPUT_DIR / 'manifest.json')

# 5. Kiểm tra kết quả
source_questions = read_records(INPUT_DIR / 'questions.jsonl', questions=True)
source_answers = read_records(INPUT_DIR / 'answers.jsonl')
saved_manifest = json.loads((OUTPUT_DIR / 'manifest.json').read_text(encoding='utf-8'))
assert saved_manifest['seed'] == SEED, 'Seed trong manifest không khớp cấu hình'
assert saved_manifest['input_dir'] == str(INPUT_DIR.expanduser().resolve()), 'Nguồn trong manifest không khớp'
seen_ids = set()
seen_businesses = set()

for subset in ('train', 'validation'):
    subset_dir = OUTPUT_DIR / subset
    questions = read_records(subset_dir / 'questions.jsonl', questions=True)
    answers = read_records(subset_dir / 'answers.jsonl')
    ids = set(questions)
    businesses = {record['business_id'] for record, _ in questions.values()}
    details = saved_manifest['splits'][subset]

    assert ids, f'{subset}: tập rỗng'
    assert not (seen_ids & ids), 'ID xuất hiện trong cả hai tập'
    assert not (seen_businesses & businesses), 'Doanh nghiệp xuất hiện trong cả hai tập'
    assert list(questions) == list(answers), f'{subset}: thứ tự đáp án không khớp'
    assert list(questions) == [record_id for record_id in source_questions if record_id in ids]
    for record_id in ids:
        assert questions[record_id] == source_questions[record_id], f'Câu hỏi thay đổi: {record_id}'
        assert answers[record_id] == source_answers[record_id], f'Đáp án thay đổi: {record_id}'

    assert len(ids) == details['pair_count']
    assert len(businesses) == details['business_count']
    assert businesses == set(details['business_ids'])
    assert {path.name for path in (subset_dir / 'data').iterdir()} == businesses
    for filename in SHARED_FILES:
        assert (subset_dir / filename).read_bytes() == (INPUT_DIR / filename).read_bytes()
    for business in businesses:
        source_business = INPUT_DIR / 'data' / business
        copied_business = subset_dir / 'data' / business
        expected = {path.relative_to(source_business) for path in source_business.rglob('*') if path.is_file()}
        actual = {path.relative_to(copied_business) for path in copied_business.rglob('*') if path.is_file()}
        assert expected == actual, f'Thiếu/thừa file nghiệp vụ: {business}'
        for relative in expected:
            assert (source_business / relative).stat().st_size == (copied_business / relative).stat().st_size
    seen_ids.update(ids)
    seen_businesses.update(businesses)

assert seen_ids == set(source_questions) == set(source_answers)
assert seen_businesses == {record['business_id'] for record, _ in source_questions.values()}
assert len(seen_ids) == saved_manifest['total_pairs']
assert len(seen_businesses) == saved_manifest['total_businesses']
assert saved_manifest['actual_validation_pair_fraction'] == saved_manifest['splits']['validation']['pair_count'] / len(seen_ids)
print('Kiểm tra đạt: đủ dữ liệu, không trùng doanh nghiệp, câu hỏi/đáp án khớp và file hỗ trợ đầy đủ.')
