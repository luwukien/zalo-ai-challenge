"""Refresh notebook function cells in an existing Kaggle kernel; keep its model.

Only fetch code from the project's authorized repository. This is trusted-code
execution, not a sandbox. No polling thread and no inference are started here.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO = "luwukien/zalo-ai-challenge"
BRANCH = "codex/datamate-baseline"
NOTEBOOK_PATH = "notebooks/datamate_qwen_kaggle.ipynb"


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "datamate-kaggle-sync", "Cache-Control": "no-cache"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def definition_source(notebook):
    """Reject setup/execution cells, including unexpected top-level actions."""
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or "definitions" not in cell.get("metadata", {}).get("tags", []):
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                continue
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id in {"BASE_SYSTEM", "PYTHON_CONTRACT"}
                for target in node.targets
            ):
                continue
            raise ValueError("Definition cell contains setup/execution code; update needs manual review.")
        sources.append(source)
    if not sources:
        raise ValueError("No definition cells found; existing session was not changed.")
    return "\n\n".join(sources)


def sync_notebook(namespace):
    """Call sync_notebook(globals()) from an idle notebook cell.

    Download/compile/stage definitions before replacing existing functions. Never
    run configuration, pip, model loading, inference, or evaluation notebook cells.
    """
    if "dm" not in namespace:
        raise RuntimeError("Chạy cell cài thư viện/import datamate_baseline trước khi cập nhật.")
    commit = fetch_json(f"https://api.github.com/repos/{REPO}/commits/{quote(BRANCH, safe='')}")["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid commit SHA from GitHub.")
    notebook = fetch_json(f"https://raw.githubusercontent.com/{REPO}/{commit}/{NOTEBOOK_PATH}")
    source = definition_source(notebook)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if namespace.get("DATAMATE_DEFINITIONS_SHA256") == digest:
        print(f"Code đang là bản mới nhất ({commit[:7]}), không thay đổi state.")
        return commit

    compiled = compile(source, f"github:{commit}/{NOTEBOOK_PATH}", "exec")
    # Validate all definitions without changing the live globals on failure.
    staged = dict(namespace)
    exec(compiled, staged)
    names = []
    for node in ast.parse(source).body:
        names.extend([node.name] if isinstance(node, ast.FunctionDef)
                     else [target.id for target in node.targets])

    output = namespace.get("OUTPUT_DIR")
    if output is not None:
        snapshots = Path(output) / "code_updates"
        snapshots.mkdir(parents=True, exist_ok=True)
        (snapshots / f"{commit}.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # Execute again with live globals so functions see later cell assignments.
    previous = {name: namespace[name] for name in names if name in namespace}
    try:
        exec(compiled, namespace)
    except Exception:
        for name in names:
            namespace.pop(name, None)
        namespace.update(previous)
        raise
    namespace["DATAMATE_CODE_SHA"] = commit
    namespace["DATAMATE_DEFINITIONS_SHA256"] = digest
    state = namespace.get("state")
    if isinstance(state, dict) and "trace" in state:
        state["trace"].setdefault("code_updates", []).append({"commit": commit, "sha256": digest})
        namespace["invalidate"](state, "knowledge", "insights", "probe", "plan")
    print(f"Đã cập nhật {len(names)} hàm/hằng từ {commit[:7]}; giữ MODEL, dữ liệu và cấu hình.")
    print("Chạy lại từ bước 3. Không chạy lại các cell định nghĩa cũ hoặc cell nạp model.")
    return commit
