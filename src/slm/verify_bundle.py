"""Check the data bundle against its MANIFEST.sha256 and the frozen test set.

Lab steps (docs/gpu_setup.md), from the repo root:
    mkdir -p bundle && tar -xf slm_bundle.tar -C bundle
    python -m src.slm.verify_bundle --root bundle --install

1. Every file listed in bundle/MANIFEST.sha256 must exist with that SHA-256.
2. The bundle's test_ids.csv must have the SHA-256 logged in decisions_log.md.
3. With --install, the data files are copied into data/processed/, and
   verify_test_checksum() is then called on the installed copy.
4. Bundled configs/docs are compared with the repo's (from git); differences are reported,
   not copied: the repo is the source of truth for code, configs and docs.
Exits non-zero on any missing file or mismatch.
"""
import hashlib
import pathlib
import shutil
import sys

from src.slm import common


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = common.parser(__doc__, device=False)
    p.add_argument("--root", default=str(common.ROOT / "bundle"),
                   help="folder the bundle was unpacked in")
    p.add_argument("--install", action="store_true",
                   help="copy the verified data files into --target")
    p.add_argument("--target", default=str(common.ROOT / "data/processed"),
                   help="where --install copies the data files (default data/processed)")
    args = p.parse_args()
    log = common.setup_logging("verify_bundle", args.log)
    root = pathlib.Path(args.root)
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        log.error("manifest not found: %s", manifest)
        sys.exit(1)

    bad, files = 0, []
    for line in (l for l in manifest.read_text().splitlines() if l.strip()):
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        files.append(name)
        path = root / name
        if not path.exists():
            log.error("MISSING  %s", name)
            bad += 1
        elif sha256(path) != expected:
            log.error("MISMATCH %s: expected %s, got %s", name, expected, sha256(path))
            bad += 1
        else:
            log.info("OK       %s", name)
    extra = sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()
                   and str(f.relative_to(root)) not in files + ["MANIFEST.sha256"])
    if extra:
        log.warning("files in the bundle folder but not in the manifest: %s", extra)
    log.info("%d files in manifest, %d failed", len(files), bad)

    from src import splits
    logged = splits.logged_checksum()
    bundled = root / "data/processed/test_ids.csv"
    if logged is None:
        log.error("no test_ids.csv SHA-256 in decisions_log.md (run from the repo root)")
        bad += 1
    elif not bundled.exists() or sha256(bundled) != logged:
        log.error("bundled test_ids.csv does not match the logged SHA-256 %s", logged)
        bad += 1
    else:
        log.info("bundled test_ids.csv matches the logged SHA-256 %s", logged)

    for name in files:
        if name.startswith(("configs/", "docs/")):
            repo = common.ROOT / name
            if not repo.exists():
                log.info("bundled %s is not in this checkout", name)
            elif sha256(repo) != sha256(root / name):
                log.info("bundled %s differs from the repo copy (repo is used)", name)

    if bad:
        log.error("BUNDLE CHECK FAILED")
        sys.exit(1)
    if args.install:
        target = pathlib.Path(args.target)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.startswith("data/processed/"):
                shutil.copy2(root / name, target / pathlib.Path(name).name)
                log.info("installed %s", target / pathlib.Path(name).name)
        if sha256(target / "test_ids.csv") != logged:
            log.error("installed test_ids.csv does not match the logged SHA-256")
            sys.exit(1)
        if target.resolve() == splits.TEST_IDS.parent.resolve():
            try:
                log.info("verify_test_checksum(): %s", splits.verify_test_checksum())
            except Exception as e:
                log.error("verify_test_checksum() failed after install: %s", e)
                sys.exit(1)
    log.info("BUNDLE OK")


if __name__ == "__main__":
    main()
