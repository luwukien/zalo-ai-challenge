import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import datamate_baseline as dm
from scripts import datamate_kaggle_sync as sync


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((Path(__file__).resolve().parents[1] /
            sync.NOTEBOOK_PATH).read_text(encoding="utf-8"))
        self.commit = "a" * 40

    def test_updates_definitions_preserves_model_config_and_live_globals(self):
        with tempfile.TemporaryDirectory() as folder:
            model, config = object(), object()
            namespace = {"dm": dm, "MODEL": model, "config": config, "OUTPUT_DIR": Path(folder)}
            with patch.object(sync, "fetch_json", side_effect=[{"sha": self.commit}, self.notebook]):
                sync.sync_notebook(namespace)
            self.assertIs(namespace["MODEL"], model)
            self.assertIs(namespace["config"], config)
            self.assertIs(namespace["ask"].__globals__, namespace)
            self.assertNotIn("questions", namespace)  # No data loading or batch cells executed.
            self.assertTrue((Path(folder) / "code_updates" / f"{self.commit}.ipynb").exists())
            original = namespace["ask"]
            with patch.object(sync, "fetch_json", side_effect=[{"sha": self.commit}, self.notebook]):
                sync.sync_notebook(namespace)
            self.assertIs(namespace["ask"], original)

    def test_network_or_invalid_code_failure_keeps_old_functions(self):
        original = object()
        namespace = {"dm": dm, "ask": original}
        with patch.object(sync, "fetch_json", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                sync.sync_notebook(namespace)
        self.assertIs(namespace["ask"], original)
        invalid = {"cells": [{"cell_type": "code", "metadata": {"tags": ["definitions"]},
                               "source": ["MODEL = None\n"]}]}
        with patch.object(sync, "fetch_json", side_effect=[{"sha": self.commit}, invalid]):
            with self.assertRaises(ValueError):
                sync.sync_notebook(namespace)
        self.assertIs(namespace["ask"], original)

    def test_sync_invalidates_old_results_but_preserves_trace_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = {"folder": root, "knowledge": {}, "plan": {},
                     "trace": {"events": [{"stage": "old"}], "prediction": {"answer": 0}}}
            namespace = {"dm": dm, "state": state}
            with patch.object(sync, "fetch_json", side_effect=[{"sha": self.commit}, self.notebook]):
                sync.sync_notebook(namespace)
            self.assertNotIn("plan", state)
            self.assertNotIn("prediction", state["trace"])
            self.assertEqual(state["trace"]["events"], [{"stage": "old"}])
            self.assertEqual(state["trace"]["code_updates"][0]["commit"], self.commit)
            self.assertTrue((root / "trace.json").exists())
