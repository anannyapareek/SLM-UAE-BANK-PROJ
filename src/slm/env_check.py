"""Report the machine: torch, CUDA, driver, GPU, VRAM, disk, bf16, bitsandbytes.

Detects and reports; never assumes. Anything missing is recorded as such, and the script
still writes its report. Exit code 1 only if the requested --device is not usable.

Run: python -m src.slm.env_check --device cuda
Writes results/gpu_env.json.
"""
import importlib
import shutil
import subprocess

from src.slm import common


def version(module):
    try:
        return importlib.import_module(module).__version__
    except Exception as e:  # missing or broken install: report it
        return f"not available ({type(e).__name__}: {e})"


def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    query = "name,driver_version,memory.total,memory.free,compute_cap"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                         capture_output=True, text=True)
    head = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda_line = next((l for l in head.splitlines() if "CUDA Version" in l), "")
    return {"available": out.returncode == 0, "gpus": out.stdout.strip().splitlines(),
            "driver_cuda_line": cuda_line.strip(" |"), "error": out.stderr.strip() or None}


def main():
    p = common.parser(__doc__)
    p.add_argument("--out", default=str(common.ROOT / "results/gpu_env.json"))
    args = p.parse_args()
    log = common.setup_logging("env_check", args.log)

    env = {"python": __import__("platform").python_version(),
           "platform": __import__("platform").platform(),
           "requested_device": args.device,
           "packages": {m: version(m) for m in ("torch", "transformers", "peft", "accelerate",
                                                "trl", "bitsandbytes", "pandas", "pyarrow",
                                                "sklearn", "yaml", "duckdb")},
           "nvidia_smi": nvidia_smi()}
    total, used, free = shutil.disk_usage(common.ROOT)
    env["disk"] = {"path": str(common.ROOT), "free_gb": round(free / 1e9, 1),
                   "total_gb": round(total / 1e9, 1)}
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        env["cpu_logical_cores"] = psutil.cpu_count()
    except Exception as e:
        env["ram_gb"] = f"not available ({e})"

    usable = False
    try:
        import torch
        env["torch_cuda_build"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        env["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                    and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            env["cudnn"] = torch.backends.cudnn.version()
            env["gpus"] = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                free_b, total_b = torch.cuda.mem_get_info(i)
                env["gpus"].append({"index": i, "name": prop.name,
                                    "vram_total_gb": round(total_b / 1e9, 2),
                                    "vram_free_gb": round(free_b / 1e9, 2),
                                    "compute_capability": f"{prop.major}.{prop.minor}"})
            env["bf16_supported"] = torch.cuda.is_bf16_supported()
        elif env["mps_available"] and common.probe("mps")[0]:
            env["bf16_supported"] = common.mps_supports_bf16()
        for name in ("cuda", "mps"):
            if env[f"{name}_available"]:
                works, detail = common.probe(name)
                env[f"{name}_probe"] = detail
                env[f"{name}_available"] = works
        usable = {"cuda": env["cuda_available"], "mps": env["mps_available"],
                  "cpu": True}[args.device]
    except Exception as e:
        env["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import bitsandbytes  # noqa: F401
        env["bitsandbytes"] = {"importable": True, "version": bitsandbytes.__version__}
    except Exception as e:
        env["bitsandbytes"] = {"importable": False, "error": f"{type(e).__name__}: {e}"}

    vram = max((g["vram_total_gb"] for g in env.get("gpus", [])), default=None)
    limit = common.config()["lora"]["qlora_below_gb"]
    env["training_mode"] = ("n/a (no CUDA GPU)" if vram is None else
                            "qlora (4-bit)" if vram < limit else "lora (bf16)")
    env["device_usable"] = usable

    for key, value in env.items():
        log.info("%s: %s", key, value)
    common.write_json(args.out, env)
    log.info("wrote %s", args.out)
    if not usable:
        raise SystemExit(f"--device {args.device} is not usable on this machine")


if __name__ == "__main__":
    main()
