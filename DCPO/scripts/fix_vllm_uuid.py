"""Patch an older vLLM in the *active Python environment* to resolve GPU UUIDs.

Run with that environment's Python. Back up cuda.py before changing it.
This patch applies only to the older vLLM implementation that contains the
specific integer conversion seen in the Hopper traceback.
"""

from pathlib import Path
import shutil
import sysconfig


cuda_file = Path(sysconfig.get_paths()["purelib"]) / "vllm/platforms/cuda.py"
backup = cuda_file.with_name("cuda.py.before_gpu_uuid_fix")

if not cuda_file.is_file():
    raise SystemExit(f"Cannot find vLLM CUDA source: {cuda_file}")

source = cuda_file.read_text(encoding="utf-8")
old = "        return int(physical_device_id)"
new = """        try:
            return int(physical_device_id)
        except ValueError:
            if not physical_device_id.startswith("GPU-"):
                raise
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                return pynvml.nvmlDeviceGetIndex(handle)
            finally:
                pynvml.nvmlShutdown()"""

if "nvmlDeviceGetHandleByUUID(physical_device_id)" in source:
    print(f"UUID support already present: {cuda_file}")
elif source.count(old) != 1 or "pynvml" not in source:
    raise SystemExit(
        "Installed vLLM does not match the affected version; file unchanged. "
        f"Inspect {cuda_file} before modifying it."
    )
elif backup.exists():
    raise SystemExit(f"Backup already exists; file unchanged: {backup}")
else:
    updated = source.replace(old, new, 1)
    compile(updated, str(cuda_file), "exec")
    shutil.copy2(cuda_file, backup)
    cuda_file.write_text(updated, encoding="utf-8")
    print(f"Patched: {cuda_file}")
    print(f"Backup:  {backup}")
