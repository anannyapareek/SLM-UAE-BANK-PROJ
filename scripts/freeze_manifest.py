"""Freeze manifest (S8-4): SHA-256 of every file that feeds a reported number.

  python -m scripts.freeze_manifest build    writes results/freeze/MANIFEST.sha256
  python -m scripts.freeze_manifest verify   checks every listed file that exists here
                                             (the lab PC holds only a subset); exit 1 on a mismatch

Covered: everything under results/ (stored predictions, eval JSONs, model files, reports,
rationales.parquet, Step 8 outputs) except results/freeze/ itself; the processed data and the
frozen test IDs (data/processed/); the Step 8 inputs (data/step8/); requirements-mac.txt and
requirements-gpu.txt. Not covered (not on the Mac): raw CERT data (data/raw/, verified at Gate 1),
the lab model weights and LoRA adapters (runs/, models/ on the lab PC).
Format: "<sha256>  <path>" per line, sorted by path, paths relative to the repo root.
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path.cwd()
OUT = ROOT / "results/freeze/MANIFEST.sha256"
GLOBS = ["results/**/*", "data/processed/*", "data/step8/*", "requirements-mac.txt", "requirements-gpu.txt"]
SKIP = ("results/freeze/", ".DS_Store")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files():
    seen = set()
    for g in GLOBS:
        for p in ROOT.glob(g):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not any(s in rel for s in SKIP):
                seen.add(rel)
    return sorted(seen)


def build():
    from src.splits import verify_test_checksum
    verify_test_checksum()
    lines = [f"{sha256(ROOT / f)}  {f}" for f in files()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(lines)} files")


def verify():
    ok = missing = bad = 0
    for line in OUT.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.exists():
            missing += 1
            continue
        if sha256(path) == digest:
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH {rel}")
    print(f"manifest: {ok} OK, {bad} mismatched, {missing} not present here")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "verify":
        sys.exit(verify())
    else:
        raise SystemExit(__doc__)
