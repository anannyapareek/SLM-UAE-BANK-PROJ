"""Step 8.2: src/figures.py and src/paper_tables.py read results/ only, and every table value equals its source.

Runs both scripts in a subprocess on a temporary copy of results/ (JSON and CSV files under 5 MB: no
predictions, no model files) with an audit hook on every file open. The run fails if either script
opens anything inside the repository other than its own Python code (data/, models/, configs/, the
real results/ ...) or anything under the home directory outside the temporary directories and the
Python / matplotlib installation. Then every cell in provenance.json is checked against its source
file, re-read independently, and against the rendered table text.
"""
import csv
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

REPO = pathlib.Path(__file__).resolve().parents[1]
MAX_BYTES = 5_000_000

RUNNER = r'''
import json, os, pathlib, sys
results, out, mpl, repo = (pathlib.Path(p).resolve() for p in sys.argv[1:5])
home = pathlib.Path.home().resolve()
runtime = {pathlib.Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
fonts = [home / "Library/Fonts"]
seen = []

def inside(p, root):
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False

def hook(event, args):
    if event != "open" or isinstance(args[0], int) or args[0] is None:
        return
    p = pathlib.Path(os.path.abspath(os.fsdecode(args[0])))
    if any(inside(p, r) for r in (results, out, mpl)) or any(inside(p, r) for r in runtime):
        return
    if inside(p, repo):
        if inside(p, repo / "src") and (p.suffix in (".py", ".pyc") or "__pycache__" in p.parts):
            return
        seen.append(str(p))
    elif inside(p, home) and not any(inside(p, f) for f in fonts):
        seen.append(str(p))

sys.path.insert(0, str(repo))
from src import figures, paper_tables
sys.addaudithook(hook)
figures.main(["--results", str(results), "--out", str(out / "figures")])
paper_tables.main(["--results", str(results), "--out", str(out / "tables")])
print("OUTSIDE " + json.dumps(sorted(set(seen))))
'''


def copy_results(dst):
    for path in (REPO / "results").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv") and path.stat().st_size < MAX_BYTES:
            target = dst / path.relative_to(REPO / "results")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("paper")
    results, out, mpl = tmp / "results", tmp / "out", tmp / "mpl"
    copy_results(results)
    out.mkdir()
    mpl.mkdir()
    env = {**os.environ, "MPLCONFIGDIR": str(mpl), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, "-c", RUNNER, str(results), str(out), str(mpl), str(REPO)],
                          cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("OUTSIDE "))
    return results, out, json.loads(line.removeprefix("OUTSIDE "))


def test_reads_only_results(built):
    _results, out, outside = built
    assert outside == [], f"files opened outside the results copy: {outside[:20]}"
    for name in ("reliability_test", "abstention_main", "abstention_appendix", "review_budget_main",
                 "review_budget_c2", "review_budget_weekday"):
        for ext in ("pdf", "png"):
            assert (out / "figures" / f"{name}.{ext}").stat().st_size > 1000
    assert (out / "figures" / "captions.md").exists()


def lookup(results, file, path):
    """Independent re-read of one source value."""
    full = results / file
    if file.endswith(".json"):
        x = json.loads(full.read_text())
    else:
        with open(full, newline="") as f:
            x = list(csv.DictReader(f))
    try:
        for k in path:
            if isinstance(k, dict):
                matches = [r for r in x if all(r[c] == v for c, v in k.items())]
                assert len(matches) == 1, f"{file}: {k} matches {len(matches)} rows"
                x = matches[0]
            else:
                x = x[k]
    except (KeyError, IndexError):
        return None
    if isinstance(x, str):
        return None if x == "" else float(x)
    return x


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_table_values_equal_sources(built):
    from src.paper_tables import render, tex_escape
    results, out, _ = built
    prov = json.loads((out / "tables" / "provenance.json").read_text())
    assert len(prov) > 400
    texts = {}
    for cell in prov:
        values = []
        for s in cell["sources"]:
            v = lookup(results, s["file"], s["path"])
            assert same(v, s["value"]), f"{cell['table']} / {cell['row']} / {cell['column']}: {s['file']} {s['path']} " \
                                        f"= {v!r}, table used {s['value']!r}"
            values.append(v)
        assert render(cell["kind"], values, cell["dp"]) == cell["text"], cell
        if cell["kind"] in ("msci", "ms", "diff"):
            assert values[0] is not None, f"missing source value: {cell}"
        name = cell["table"]
        if name not in texts:
            texts[name] = ((out / "tables" / f"{name}.md").read_text(), (out / "tables" / f"{name}.tex").read_text())
        md, tex = texts[name]
        assert cell["text"] in md and tex_escape(cell["text"]) in tex, cell


def test_committed_tables_are_current(built):
    _results, out, _ = built
    committed = REPO / "paper/tables"
    if not committed.exists():
        pytest.skip("paper/tables not built")
    for f in sorted((out / "tables").iterdir()):
        assert f.read_text() == (committed / f.name).read_text(), f"paper/tables/{f.name} is stale: rerun src.paper_tables"
