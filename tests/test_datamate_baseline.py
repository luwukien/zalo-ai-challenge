"""Synthetic fixtures only. Optional real-data inspection is documented separately."""
import ast
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from contextlib import nullcontext

import numpy as np
import openpyxl
import pandas as pd

from scripts import datamate_baseline as baseline


SELECT = json.dumps({"tables": {"sales": ["order_id", "amount"]}, "answer_type": "number"})
CODE = '''def answer(tables, as_of_date, calendar):
    total = tables["sales"]["amount"].map(parse_money).sum()
    return {"type": "number", "answer": round_half_up(total)}
'''


class FakeModel:
    info = {"backend": "fake", "model_id": "fake", "resolved_revision": "test-v1",
            "model_load_seconds": 0, "versions": {"fake": "1"}}

    def __init__(self, responses=(), tokens=100):
        self.responses = iter(responses)
        self.calls = []
        self.tokens = tokens

    def count_tokens(self, messages):
        return self.tokens

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return baseline.Generation(response, self.tokens, 20, 0.01)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="datamate-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, count=1):
        data_root = self.root / "dataset"
        split = data_root / "train"
        business = split / "data" / "biz_1"
        business.mkdir(parents=True)
        (business / "README.md").write_text("# Synthetic sales\n", encoding="utf-8")
        (business / "sales.csv").write_text("order_id,amount\n001,25k\n002,55000\n002,55000\n", encoding="utf-8")
        (split / "rulebook.md").write_text("Dữ liệu giả: giữ các dòng trùng. Hôm nay là as_of_date.", encoding="utf-8")
        (split / "calendar.csv").write_text("date,lunar_day\n2026-01-02,14\n", encoding="utf-8")
        self.questions = [{"id": f"q_{i}", "business_id": "biz_1", "question": "Tổng tiền?",
                           "as_of_date": "2026-01-02"} for i in range(count)]
        (split / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in self.questions), encoding="utf-8")
        (split / "answers.jsonl").write_text("".join(json.dumps({"id": q["id"], "type": "number",
            "answer": 135000, "alias": [], "secret": "GOLD_MUST_NEVER_ENTER_PROMPT"}) + "\n"
            for q in self.questions), encoding="utf-8")
        return baseline.Config(data_root=str(data_root), output_dir=str(self.root / "outputs"), exec_timeout=20)

    def workbook(self, directory, filename, sheets):
        book = openpyxl.Workbook()
        book.remove(book.active)
        for name, rows in sheets.items():
            sheet = book.create_sheet(name)
            for row in rows:
                sheet.append(row)
        book.save(directory / filename)
        book.close()

    def test_numbers_dates_and_rounding(self):
        for value, expected in [("55.000đ", 55000), ("38,000", 38000), ("25k", 25000),
                                ("1.234,50", 1234.5), ("1,234.50", 1234.5), ("1,5", 1.5),
                                ("(25.000 VND)", -25000), ("18,914tr", 18914000), ("-3", -3), (12.5, 12.5)]:
            with self.subTest(value=value):
                self.assertEqual(baseline.parse_number(value), expected)
        self.assertEqual(baseline.parse_number("1.234", decimal_separator="."), 1.234)
        self.assertTrue(np.isnan(baseline.parse_money("")))
        for value in ("unknown", True, "1..4", float("inf")):
            with self.assertRaises(ValueError):
                baseline.parse_number(value)
        self.assertEqual(baseline.round_half_up(2.5), 3)
        self.assertEqual(baseline.round_half_up(12.345, 2), 12.35)
        self.assertEqual(baseline.parse_date("01/02/2025"), pd.Timestamp("2025-02-01"))
        self.assertEqual(baseline.parse_date(45689), pd.Timestamp("2025-02-01"))
        self.assertEqual(baseline.parse_date("01/02", sheet="Thang 02 2025"), pd.Timestamp("2025-02-01"))
        self.assertEqual(baseline.parse_date(dt.datetime(2025, 2, 1)), pd.Timestamp("2025-02-01"))
        with self.assertRaises(ValueError):
            baseline.parse_date("01/02")

    def test_excel_headers_partitions_and_sources(self):
        directory = self.root / "biz"
        directory.mkdir()
        (directory / "README.md").write_text(
            "Mỗi tháng một sheet; tách thành nhiều file.\n"
            "## Cột trong `Sales_1.xlsx`\n- **Ngày**: ngày\n- **Amount**: tiền\n"
            "## Cột trong `Sales_2026.xlsx`\n- **Ngày**: ngày\n- **Amount**: tiền\n",
            encoding="utf-8")
        self.workbook(directory, "Sales_1.xlsx", {
            "T01_2025": [["Nhóm", None], ["Ngày", "Amount"], ["01/01", "25k"], ["01/01", "25k"]],
            "02.2025": [["Ngày", "Amount"], [45689, "55.000đ"]]})
        self.workbook(directory, "Sales_2026.xlsx", {"T01_2026": [["Ngày", "Amount"], ["01/01", "30k"]]})
        self.workbook(directory, "Catalog.xlsx", {"Products": [["TITLE"], [], ["name", "price"], ["Tea", 5]],
                                                  "Units": [["name", "factor"], ["Tea", 10]]})
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        tables, sources, _ = baseline.load_business(directory)
        self.assertEqual(set(tables), {"Sales", "Catalog::Products", "Catalog::Units"})
        self.assertEqual(len(tables["Sales"]), 4)
        self.assertEqual(sources["Sales"][0]["header_row"], 2)
        self.assertEqual(tables["Sales"]["__source_row"].tolist(), [3, 4, 2, 2])
        self.assertEqual(tables["Sales"]["Amount"].tolist(), ["25k", "25k", "55.000đ", "30k"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_csv_and_readonly_sqlite_preserve_rows(self):
        directory = self.root / "biz"
        directory.mkdir()
        (directory / "README.md").write_text("Synthetic database. No partitions.", encoding="utf-8")
        (directory / "sales.csv").write_text("id,amount\n001,25k\n001,25k\n", encoding="utf-8")
        db = directory / "pos.sqlite"
        with sqlite3.connect(db) as connection:
            connection.execute('CREATE TABLE "order details" (id TEXT, amount REAL)')
            connection.executemany('INSERT INTO "order details" VALUES (?, ?)', [("001", 25), ("001", 25)])
        connection.close()
        before = db.read_bytes()
        tables, _, _ = baseline.load_business(directory)
        self.assertEqual(tables["sales"]["id"].tolist(), ["001", "001"])
        self.assertEqual(len(tables["pos::order details"]), 2)
        self.assertEqual(before, db.read_bytes())

    def test_worker_join_and_full_rows(self):
        tables = {"lines": pd.DataFrame({"id": [1, 1, 2], "amount": [10, 15, 100]}),
                  "orders": pd.DataFrame({"id": [1, 2], "status": ["done", "cancel"]})}
        code = '''def answer(tables, as_of_date, calendar):
    df = tables["lines"].merge(tables["orders"], on="id", validate="many_to_one")
    return {"type": "number", "answer": df.loc[df["status"] == "done", "amount"].sum()}
'''
        result = baseline.execute_code(code, tables, "2026-01-02", pd.DataFrame(), timeout=20)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["prediction"]["answer"], 25)

    def test_worker_timeout_and_io_rejected(self):
        code = "def answer(tables, as_of_date, calendar):\n    while True:\n        pass\n"
        result = baseline.execute_code(code, {}, "2026-01-02", pd.DataFrame(), timeout=1)
        self.assertFalse(result["ok"])
        self.assertIn("ExecutionTimeout", result["error"])
        for body in ("import os\n    os.system('echo hi')", "return pd.read_csv('data.csv')",
                     "return tables['x'].to_csv('data.csv')", "return getattr(pd, 'read_csv')('x')",
                     "from numpy import load\n    return load('x')"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                baseline.validate_code("def answer(tables, as_of_date, calendar):\n    " + body)

    def test_answer_schema(self):
        for value in ({"type": "number", "answer": "12"}, {"type": "number", "answer": True},
                      {"type": "percent", "answer": float("nan")}, {"type": "boolean", "answer": 1},
                      {"type": "table", "answer": {"columns": ["name"], "rows": [["x", 1]]}},
                      {"type": "unanswerable", "reason": 1}, {"type": "date", "answer": "2026-13"}):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                baseline.validate_answer(value)
        self.assertEqual(baseline.validate_answer({"type": "number", "answer": np.int64(12)})["answer"], 12)
        self.assertEqual(baseline.validate_answer({"type": "unanswerable"}), {"type": "unanswerable"})

    def test_pipeline_repairs_and_no_gold_in_prompts(self):
        config = self.dataset()
        broken = CODE.replace('["amount"]', '["missing"]')
        model = FakeModel([SELECT, broken, CODE])
        original_read = baseline.read_jsonl

        def checked_read(path):
            if Path(path).name == "answers.jsonl":
                self.assertTrue((Path(config.output_dir) / "predictions.jsonl").exists())
            return original_read(path)

        with patch.object(baseline, "read_jsonl", side_effect=checked_read):
            summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["reference_evaluation"]["matched"], 1)
        trace = baseline.read_jsonl(Path(config.output_dir) / "trace.jsonl")[0]
        self.assertEqual(trace["repairs_used"], 1)
        self.assertIn("KeyError", trace["attempts"][1]["execution"]["error"])
        self.assertTrue(trace["attempts"][2]["execution"]["ok"])
        self.assertGreater(trace["timing"]["total_seconds"], 0)
        self.assertTrue((Path(config.output_dir) / "code/q_0/attempt_1.py").exists())
        for messages, _ in model.calls:
            self.assertNotIn("GOLD_MUST_NEVER_ENTER_PROMPT", json.dumps(messages))

    def test_invalid_selection_can_be_repaired(self):
        config = self.dataset()
        wrong = SELECT.replace('"amount"', '"not_a_column"')
        model = FakeModel([wrong, SELECT, CODE])
        summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["success"], 1)
        self.assertTrue(model.calls[1][1]["repair"])

    def test_exhausted_repair_returns_abstain(self):
        config = replace(self.dataset(), max_repairs=1)
        model = FakeModel([SELECT, "not Python", "still not Python"])
        summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["failed"], 1)
        prediction = baseline.read_jsonl(Path(config.output_dir) / "predictions.jsonl")[0]
        self.assertEqual(prediction, {"id": "q_0", "type": "abstain"})

    def test_invalid_answer_format_enters_repair(self):
        config = self.dataset()
        invalid = "def answer(tables, as_of_date, calendar):\n    return {'type': 'number', 'answer': '135000'}\n"
        summary = baseline.run_baseline(config, model=FakeModel([SELECT, invalid, CODE]))
        self.assertEqual(summary["success"], 1)
        trace = baseline.read_jsonl(Path(config.output_dir) / "trace.jsonl")[0]
        self.assertIn("Invalid answer format", trace["attempts"][1]["execution"]["error"])

    def test_unanswerable_is_not_technical_failure(self):
        config = self.dataset()
        code = "def answer(tables, as_of_date, calendar):\n    return {'type': 'unanswerable', 'reason': 'Missing business data'}\n"
        model = FakeModel([SELECT, code])
        summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(len(model.calls), 2)

    def test_selection_json_null_and_unknown_table_rejected(self):
        tables = {"sales": pd.DataFrame({"amount": [1]})}
        for text in ('null', '{"tables": null, "answer_type": "number"}',
                     '{"tables": {"other": ["amount"]}, "answer_type": "number"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                baseline.parse_selection(text, tables)

    def test_context_overflow_never_sends_truncated_prompt(self):
        config = self.dataset()
        model = FakeModel(tokens=config.max_input_tokens + 1)
        summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(model.calls, [])
        trace = baseline.read_jsonl(Path(config.output_dir) / "trace.jsonl")[0]
        self.assertIn("ContextOverflow", trace["error"])

    def test_resume_extend_limit_and_detect_input_changes(self):
        config = replace(self.dataset(count=2), limit=1)
        baseline.run_baseline(config, model=FakeModel([SELECT, CODE]))
        resume = replace(config, limit=2, resume=True)
        model = FakeModel([SELECT, CODE])
        summary = baseline.run_baseline(resume, model=model)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(baseline.read_jsonl(Path(config.output_dir) / "predictions.jsonl")), 2)
        again = baseline.run_baseline(resume, model=FakeModel())
        self.assertEqual(again["completed"], 2)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            baseline.run_baseline(replace(resume, seed=99), model=FakeModel())
        source = Path(config.data_root) / "train/data/biz_1/sales.csv"
        source.write_text(source.read_text() + "003,1\n")
        with self.assertRaisesRegex(ValueError, "input data"):
            baseline.run_baseline(resume, model=FakeModel())

    def test_gpu_failure_stops_and_can_resume(self):
        config = self.dataset()
        model = FakeModel([SELECT, baseline.RuntimeFailure("CUDA out of memory")])
        summary = baseline.run_baseline(config, model=model)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["completed"], 0)
        trace = baseline.read_jsonl(Path(config.output_dir) / "trace.jsonl")[0]
        self.assertFalse(trace["complete"])
        self.assertIn("CUDA", trace["error"])
        resumed = baseline.run_baseline(replace(config, resume=True), model=FakeModel([SELECT, CODE]))
        self.assertEqual(resumed["success"], 1)
        resumed_trace = baseline.read_jsonl(Path(config.output_dir) / "trace.jsonl")[0]
        self.assertIn("CUDA", resumed_trace["previous_runs"][0]["error"])

    def test_output_may_not_overlap_dataset(self):
        config = self.dataset()
        with self.assertRaisesRegex(ValueError, "overlap"):
            baseline.run_baseline(replace(config, output_dir=str(Path(config.data_root) / "outputs")), FakeModel())

    def test_notebook_is_clean_and_python_cells_parse(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/datamate_baseline_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))

    def test_qwen_uses_text_chat_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return {"input_ids": "example"}
        model = baseline.QwenModel.__new__(baseline.QwenModel)
        model.processor = Tokenizer()
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(model.inputs(messages), {"input_ids": "example"})
        self.assertTrue(model.processor.kwargs["add_generation_prompt"])
        self.assertNotIn("enable_thinking", model.processor.kwargs)

    def test_qwen_generation_preserves_full_prompt_and_bounds_prefill(self):
        model = baseline.QwenModel.__new__(baseline.QwenModel)
        model.config = baseline.Config()
        model.skip_special_tokens = True
        inputs = MagicMock()
        inputs.to.return_value = inputs
        input_ids = MagicMock()
        input_ids.shape = (1, 12000)
        inputs.__getitem__.return_value = input_ids
        inputs.keys.return_value = ["input_ids"]
        model.inputs = MagicMock(return_value=inputs)
        model.model = MagicMock()
        model.model.generate.return_value = [list(range(12000)) + [7, 8]]
        model.processor = MagicMock()
        model.processor.decode.return_value = "answer"
        torch = MagicMock()
        torch.inference_mode.side_effect = nullcontext
        with patch.dict("sys.modules", {"torch": torch, "transformers": MagicMock()}):
            result = model.generate([], max_new_tokens=32, repair=False, seed=42)
        kwargs = model.model.generate.call_args.kwargs
        self.assertIs(kwargs["input_ids"], input_ids)
        self.assertEqual(kwargs["prefill_chunk_size"], 256)
        self.assertEqual(kwargs["logits_to_keep"], 1)
        self.assertEqual(kwargs["cache_implementation"], "dynamic")
        self.assertTrue(kwargs["use_cache"])
        self.assertEqual((result.input_tokens, result.output_tokens), (12000, 2))
        model.processor.decode.assert_called_once_with([7, 8], skip_special_tokens=True)

    def test_prefill_config_validation_and_gemma_unchanged(self):
        self.assertEqual(baseline.HFModel.__new__(baseline.HFModel).generation_options(), {})
        model = baseline.QwenModel.__new__(baseline.QwenModel)
        for size in (0, -1, True, 1.5):
            model.config = baseline.Config(qwen_prefill_chunk_size=size)
            with self.subTest(size=size), self.assertRaises(ValueError):
                model.generation_options()

    def test_kaggle_notebook_is_clean_and_python_cells_parse(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/datamate_qwen_kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
