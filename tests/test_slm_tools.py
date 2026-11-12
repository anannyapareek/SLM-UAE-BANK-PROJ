"""SLM tooling that runs without torch: bundle verification, config, stage hygiene."""
import ast
import hashlib
import pathlib
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT

SLM = ROOT / "src/slm"


def test_slm_config():
    cfg = yaml.safe_load((ROOT / "configs/slm.yaml").read_text())
    assert set(cfg["models"]) == {"primary", "second", "fallback"}
    for m in cfg["models"].values():
        assert len(m["revision"]) == 40
    assert cfg["prompt"]["chat_template"] is False
    assert cfg["prompt"]["suffix"] == "\nLabel:"
    assert cfg["lora"]["alpha_per_rank"] == 2 and cfg["lora"]["dropout"] == 0.05
    assert cfg["lora"]["epochs"] == 1 and cfg["lora"]["max_length"] == 512
    assert cfg["lora"]["grid"] == {"learning_rate": [1e-4, 2e-4], "rank": [8, 16]}
    assert 5 <= cfg["training"]["neg_per_pos"] <= 10


def test_slm_scripts_verify_checksum_before_reading_data():
    """Every src/slm module whose main() reads rows calls verify_test_checksum() first."""
    readers = ("load_rows", "split_counts", "build")
    for path in sorted(SLM.glob("*.py")):
        tree = ast.parse(path.read_text())
        mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
        if not mains:
            continue
        calls = [n for n in ast.walk(mains[0]) if isinstance(n, ast.Call)]
        name = lambda c: getattr(c.func, "attr", getattr(c.func, "id", ""))
        reads = [c.lineno for c in calls if name(c) in readers]
        verify = [c.lineno for c in calls if name(c) == "verify_test_checksum"]
        if reads:
            assert verify and min(verify) < min(reads), path.name


def run_verify(root, *extra):
    return subprocess.run([sys.executable, "-m", "src.slm.verify_bundle", "--root", str(root),
                           "--log", str(root / "verify.log"), *extra],
                          cwd=ROOT, capture_output=True, text=True)


@pytest.fixture
def bundle(tmp_path):
    test_ids = ROOT / "data/processed/test_ids.csv"
    if not test_ids.exists():
        pytest.skip("test_ids.csv not present")
    files = {"data/processed/test_ids.csv": test_ids.read_bytes(),
             "data/processed/features.parquet": b"not really parquet",
             "configs/slm.yaml": (ROOT / "configs/slm.yaml").read_bytes()}
    lines = []
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_verify_bundle_passes(bundle):
    result = run_verify(bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUNDLE OK" in result.stdout


def test_verify_bundle_fails_on_mismatch(bundle):
    (bundle / "data/processed/features.parquet").write_bytes(b"tampered")
    result = run_verify(bundle)
    assert result.returncode == 1 and "MISMATCH" in result.stdout


def test_verify_bundle_fails_on_missing(bundle):
    (bundle / "configs/slm.yaml").unlink()
    result = run_verify(bundle)
    assert result.returncode == 1 and "MISSING" in result.stdout


def test_verify_bundle_fails_on_wrong_test_ids(bundle):
    data = b"row_id\nfake\n"
    (bundle / "data/processed/test_ids.csv").write_bytes(data)
    manifest = bundle / "MANIFEST.sha256"
    lines = [l for l in manifest.read_text().splitlines() if "test_ids" not in l]
    manifest.write_text("\n".join(lines + [f"{hashlib.sha256(data).hexdigest()}  "
                                           "data/processed/test_ids.csv"]) + "\n")
    result = run_verify(bundle)
    assert result.returncode == 1 and "does not match the logged" in result.stdout


def test_resume_refuses_a_different_seed(tmp_path):
    from src.slm.train_lora import check_run_args
    run = {"model": "primary", "lr": 1e-4, "rank": 8, "seed": 2}
    check_run_args(tmp_path, run, resuming=False)
    check_run_args(tmp_path, run, resuming=True)                     # same run: resumes
    with pytest.raises(SystemExit, match="different run"):
        check_run_args(tmp_path, dict(run, seed=3), resuming=True)
    check_run_args(tmp_path, dict(run, seed=3), resuming=False)      # fresh start is allowed
