"""Fetch the SLM weights and tokenizers pinned in configs/slm.yaml into a local folder.

Default: download from Hugging Face at the pinned revision (uses HF_TOKEN from the
environment if set; the chosen models are not gated, so it is optional).
--from-dir DIR: copy from a folder fetched another way (e.g. Google Drive) that has the same
layout (<DIR>/<org>__<name>/...), instead of downloading.

Either way it writes <out>/download_record.json with each model's revision, date and the
SHA-256 of every file, and checks that each folder has a config, a tokenizer and weights.

Run from the repo root:
    python scripts/download_models.py [--models primary second fallback] [--out models]
    python scripts/download_models.py --from-dir /path/to/drive/models
Then pass --models-dir models to the src/slm scripts.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_folder(folder):
    names = {p.name for p in folder.iterdir()}
    problems = []
    if "config.json" not in names:
        problems.append("config.json")
    if not names & {"tokenizer.json", "tokenizer.model", "vocab.json"}:
        problems.append("tokenizer")
    if not any(n.endswith(".safetensors") for n in names):
        problems.append("*.safetensors")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["primary", "second", "fallback"])
    p.add_argument("--out", default=str(ROOT / "models"))
    p.add_argument("--from-dir", help="copy from this folder instead of downloading")
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/slm.yaml").read_text())["models"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "download_record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    failed = False

    for role in args.models:
        model_id, revision = cfg[role]["id"], cfg[role]["revision"]
        folder = out / model_id.replace("/", "__")
        print(f"== {role}: {model_id} @ {revision}", flush=True)
        if args.from_dir:
            source = pathlib.Path(args.from_dir) / folder.name
            if not source.exists():
                print(f"   not found: {source}", flush=True)
                failed = True
                continue
            if source.resolve() != folder.resolve():
                shutil.copytree(source, folder, dirs_exist_ok=True)
            how = f"copied from {source}"
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, revision=revision, local_dir=folder,
                              token=os.environ.get("HF_TOKEN") or None,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                              "*.jinja", "LICENSE*", "README.md"])
            how = "downloaded from Hugging Face"
        problems = check_folder(folder)
        if problems:
            print(f"   incomplete, missing: {problems}", flush=True)
            failed = True
        files = {str(f.relative_to(folder)): sha256(f) for f in sorted(folder.rglob("*"))
                 if f.is_file() and ".cache" not in f.parts}
        size = sum((folder / f).stat().st_size for f in files)
        record[role] = {"id": model_id, "revision": revision, "folder": str(folder),
                        "how": how, "date": datetime.date.today().isoformat(),
                        "bytes": size, "files": files}
        print(f"   {how}; {len(files)} files, {size / 1e9:.2f} GB", flush=True)

    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {record_path}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
