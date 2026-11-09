"""Shared helpers for the SLM scripts: config, logging, device, data and model loading.

Every script detects the machine rather than assuming it: the device comes from --device,
and anything the machine lacks (CUDA, bf16, bitsandbytes) is reported, not presumed.
"""
import argparse
import datetime
import json
import logging
import os
import pathlib
import platform
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/slm.yaml"
P = pathlib.Path(os.environ.get("PROCESSED_DIR", ROOT / "data/processed"))
SPLITS = ("train", "valid", "calib", "test")


def config():
    return yaml.safe_load(CONFIG.read_text())


def parser(description, device=True):
    p = argparse.ArgumentParser(description=description)
    if device:
        p.add_argument("--device", choices=["cuda", "mps", "cpu"], required=True)
    p.add_argument("--log", help="log file (default: logs/<script>_<time>.log)")
    return p


def setup_logging(name, path=None):
    """Log to stdout and a file, flushed per line, so nohup/tmux runs keep a record."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(path or ROOT / "logs" / f"{name}_{stamp}.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.info("log file %s; python %s; %s; argv %s", path, platform.python_version(),
             platform.platform(), " ".join(sys.argv))
    return log


PROBE = """
import torch
for dtype in (torch.float32, torch.bfloat16):
    x = torch.randn(4, 8, device="{d}", dtype=dtype)
    w = torch.randn(16, 8, device="{d}", dtype=dtype)
    torch.nn.functional.linear(x, w).sum().item()
print("ok")
"""


def probe(name, timeout=120):
    """Run a small matmul on the device in a child process: a broken backend can abort the
    process outright (e.g. a torch MPS build newer than macOS supports), so it is tested
    out of process. Returns (works, detail)."""
    import subprocess
    try:
        out = subprocess.run([sys.executable, "-c", PROBE.format(d=name)], capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout} s"
    if out.returncode == 0 and out.stdout.strip().endswith("ok"):
        return True, "ok"
    reason = next((l for l in out.stderr.splitlines() if "reason:" in l or "Error" in l),
                  out.stderr.strip()[-300:])
    return False, f"exit {out.returncode}: {reason.strip()[:300]}"


def device(name, log=None):
    """The torch device for --device, or exit with a clear message if it is not usable."""
    import torch
    present = {"cuda": torch.cuda.is_available(),
               "mps": getattr(torch.backends, "mps", None) is not None
               and torch.backends.mps.is_available(),
               "cpu": True}[name]
    if not present:
        raise SystemExit(f"--device {name} requested but not available on this machine")
    if name != "cpu":
        works, detail = probe(name)
        if not works:
            raise SystemExit(f"--device {name} is present but a test matmul fails: {detail}")
    if log:
        log.info("device %s (probe ok)", name)
    return torch.device(name)


def dtype(device_name):
    """bf16 where the device supports it; otherwise fp32 (reported, never assumed)."""
    import torch
    if device_name == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if device_name == "mps":
        return torch.bfloat16 if mps_supports_bf16() else torch.float32
    return torch.bfloat16


def mps_supports_bf16():
    import torch
    try:
        torch.ones(2, dtype=torch.bfloat16, device="mps") @ torch.ones(2, dtype=torch.bfloat16,
                                                                         device="mps")
        return True
    except Exception:
        return False


def sync(device_name):
    import torch
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def model_entry(name):
    """name is a role in configs/slm.yaml (primary, second, fallback) or a model path/id."""
    models = config()["models"]
    if name in models:
        return models[name]["id"], models[name]["revision"]
    return name, None


def model_source(name, models_dir=None):
    """Load from a local copy (scripts/download_models.py layout) if there is one."""
    model_id, revision = model_entry(name)
    if models_dir:
        local = pathlib.Path(models_dir) / model_id.replace("/", "__")
        if local.exists():
            return str(local), None
    if pathlib.Path(model_id).exists():
        return model_id, None
    return model_id, revision


def load_tokenizer(name, models_dir=None):
    from transformers import AutoTokenizer
    source, revision = model_source(name, models_dir)
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name, device_name, models_dir=None, quantize_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM
    source, revision = model_source(name, models_dir)
    kwargs = {"revision": revision, "dtype": dtype(device_name)}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if not quantize_4bit:
        model.to(device_name)
    return model


def answer_ids(tok):
    """Token ids of " yes" and " no" (the next token after "Label:"); each must be one token."""
    ids = {}
    for word in config()["prompt"]["label_words"]:
        encoded = tok.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"{word!r} is {len(encoded)} tokens for this tokenizer: {encoded}")
        ids[word.strip()] = encoded[0]
    return ids["yes"], ids["no"]


def load_rows(split, n=None, seed=1, allow_test=False, columns=None, only_final=False):
    """Feature rows of one split with y, as a DataFrame sorted by row_id.

    Model stage: reads the split's labels (y) like the XGBoost stage. Test needs allow_test.
    """
    import duckdb
    if split == "test" and not allow_test:
        raise SystemExit("refusing to load test rows without --allow-test")
    con = duckdb.connect(config={"threads": 1})
    select = "f.*" if columns is None else ", ".join(f'f."{c}"' for c in columns)
    where = f"x.split = '{split}'" + (" AND f.is_final" if only_final else "")
    query = f"""
        SELECT {select}, l.y FROM '{P}/features.parquet' f
        JOIN '{P}/splits.parquet' x USING (row_id)
        JOIN '{P}/labels.parquet' l USING (row_id)
        WHERE {where} ORDER BY f.row_id"""
    frame = con.execute(query).df()
    if n is not None and n < len(frame):
        frame = frame.sample(n=n, random_state=seed).sort_values("row_id")
    return frame.reset_index(drop=True)


def split_counts():
    """Rows per split, all rows and final (logoff) rows only; no labels read."""
    import duckdb
    con = duckdb.connect(config={"threads": 1})
    out = con.execute(f"""
        SELECT x.split, count(*), count(*) FILTER (f.is_final)
        FROM '{P}/splits.parquet' x JOIN '{P}/features.parquet' f USING (row_id)
        GROUP BY 1""").fetchall()
    return {s: {"hourly": a, "logoff_only": b} for s, a, b in out}


def write_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
