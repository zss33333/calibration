"""Set per-rank scratch directories, then execute the standard RLCR runner."""

import os
from pathlib import Path
import sys


def main():
    root = os.environ.get("RLCR_RANK_RUNTIME_ROOT")
    rank = os.environ.get("LOCAL_RANK")
    if not root or rank is None or not rank.isdecimal():
        raise RuntimeError("Hopper runner requires RLCR_RANK_RUNTIME_ROOT and LOCAL_RANK")

    rank_dir = Path(root) / f"rank-{rank}"
    triton_dir = rank_dir / "triton"
    inductor_dir = rank_dir / "inductor"
    tmp_dir = rank_dir / "tmp"
    triton_dir.mkdir(parents=True, exist_ok=True)
    inductor_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TMPDIR"] = str(tmp_dir)
    ray_root = os.environ.get("RLCR_RAY_TMP_ROOT")
    if ray_root:
        ray_dir = Path(f"{ray_root}_{rank}")
        ray_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAY_TMPDIR"] = str(ray_dir)
        print(f"RLCR rank {rank} Ray temp: {ray_dir}", flush=True)
    print(f"RLCR rank {rank} Triton cache: {triton_dir}", flush=True)
    runner = Path(__file__).with_name("official_upstream") / "rl_runner.py"
    os.execv(sys.executable, [sys.executable, str(runner), *sys.argv[1:]])


if __name__ == "__main__":
    main()
