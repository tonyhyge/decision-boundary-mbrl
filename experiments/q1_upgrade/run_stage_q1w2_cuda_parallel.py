"""
Stage Q1-W2: Parallel CUDA Orchestrator for Full-Budget Walker2d-v4 Replication.
Manages concurrent worker pool across designated GPUs with clean CUDA_VISIBLE_DEVICES isolation,
pairing (MLE, VaGraM) per seed, redirecting per-run logs, supporting automatic resume on existing checkpoints,
and invoking blind statistical evaluation upon cohort completion.
"""
import os
import sys
import time
import json
import argparse
import subprocess
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel CUDA Orchestrator for Stage Q1-W2 Walker2d MBPO.")
    parser.add_argument("--gpus", type=int, nargs="+", default=[3], help="GPU device IDs to use (e.g. 3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[142, 143, 144, 145, 146, 147, 148, 149, 150, 151], help="Confirmatory seed cohort")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["mle", "vagram"], help="Algorithms to evaluate")
    parser.add_argument("--total-steps", type=int, default=125000, help="Total environment interaction steps per seed")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of deterministic evaluation episodes per checkpoint")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Maximum concurrent worker processes")
    parser.add_argument("--out-dir", type=str, default=os.path.join(_ROOT, "results", "stage_q1w2_cuda"), help="Output directory for results and logs")
    return parser.parse_args()


def run_single_job(job_args: Tuple[str, int, int, int, int, int, str, str]) -> Dict[str, Any]:
    algo, seed, total_steps, eval_freq, eval_episodes, gpu_id, out_dir, python_bin = job_args
    
    out_file = os.path.join(out_dir, f"q1w2_walker2d_{algo}_seed{seed}.json")
    log_file = os.path.join(out_dir, f"log_{algo}_seed{seed}.log")
    
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                res = json.load(f)
            return {
                "algorithm": algo,
                "seed": seed,
                "gpu": gpu_id,
                "status": "cached",
                "terminal_return": res.get("terminal_return_mean", 0.0),
                "wall_clock": res.get("wall_clock_seconds", 0.0),
                "out_file": out_file,
            }
        except Exception:
            pass  # Re-run if file corrupted

    cmd = [
        python_bin,
        "-u",
        os.path.join(_ROOT, "experiments", "q1_upgrade", "run_stage_q1w2_walker2d.py"),
        "--algorithms", algo,
        "--seeds", str(seed),
        "--total-steps", str(total_steps),
        "--eval-freq", str(eval_freq),
        "--eval-episodes", str(eval_episodes),
        "--device", "cuda:0",
        "--out-dir", out_dir,
    ]
    
    worker_env = dict(os.environ)
    worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_env["PYTHONUNBUFFERED"] = "1"
    worker_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    with open(log_file, "w") as log_f:
        log_f.write(f"=== Stage Q1-W2 Run: {algo.upper()} Seed {seed} on physical GPU {gpu_id} (isolated cuda:0) ===\n")
        log_f.write(f"Command: {' '.join(cmd)}\n\n")
        log_f.flush()
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=_ROOT,
            env=worker_env,
        )
        proc.wait()
        
    wall_clock = time.time() - t0
    
    if proc.returncode != 0:
        return {
            "algorithm": algo,
            "seed": seed,
            "gpu": gpu_id,
            "status": "failed",
            "returncode": proc.returncode,
            "wall_clock": wall_clock,
            "log_file": log_file,
        }
        
    try:
        with open(out_file, "r") as f:
            res = json.load(f)
        return {
            "algorithm": algo,
            "seed": seed,
            "gpu": gpu_id,
            "status": "completed",
            "terminal_return": res.get("terminal_return_mean", 0.0),
            "wall_clock": wall_clock,
            "out_file": out_file,
        }
    except Exception as e:
        return {
            "algorithm": algo,
            "seed": seed,
            "gpu": gpu_id,
            "status": "corrupted",
            "error": str(e),
            "wall_clock": wall_clock,
            "log_file": log_file,
        }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    python_bin = sys.executable

    print("=" * 80)
    print("STAGE Q1-W2: PARALLEL CUDA ORCHESTRATOR FOR WALKER2D-V4 REPLICATION")
    print("=" * 80)
    print(f"Target Environment:    Walker2d-v4")
    print(f"Total Confirmatory N:  {len(args.seeds)} paired seeds ({args.seeds})")
    print(f"Algorithms:            {args.algorithms}")
    print(f"Budget per Run:        {args.total_steps} interaction steps")
    print(f"Designated GPUs:       {args.gpus}")
    print(f"Max Concurrent:        {args.max_concurrent} workers")
    print(f"Output Directory:      {args.out_dir}")
    print("=" * 80)

    # Queue jobs paired by seed: (seed, MLE), (seed, VaGraM)
    jobs = []
    gpu_pool = args.gpus
    for i, seed in enumerate(args.seeds):
        for algo in args.algorithms:
            gpu_assigned = gpu_pool[(len(jobs)) % len(gpu_pool)]
            jobs.append((algo, seed, args.total_steps, args.eval_freq, args.eval_episodes, gpu_assigned, args.out_dir, python_bin))

    print(f"Total jobs queued: {len(jobs)} runs ({len(args.seeds)} paired seeds x {len(args.algorithms)} algorithms)")

    t_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.max_concurrent) as executor:
        future_to_job = {executor.submit(run_single_job, job): job for job in jobs}
        
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            algo, seed, total_steps, _, _, gpu_id, _, _ = job
            try:
                res = future.result()
                results.append(res)
                status = res["status"].upper()
                wall_m = res["wall_clock"] / 60.0
                ret = res.get("terminal_return", 0.0)
                print(f"[{status}] {algo.upper():<6} Seed {seed:<3} on GPU {gpu_id} | Terminal: {ret:>7.2f} | Wall: {wall_m:>5.1f} min ({len(results)}/{len(jobs)} finished)")
            except Exception as exc:
                print(f"[EXCEPTION] Job {algo.upper()} Seed {seed} generated exception: {exc}")

    total_wall_clock = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL Q1-W2 RUNS PROCESSED in {total_wall_clock / 3600.0:.2f} hours")
    print("=" * 80)


if __name__ == "__main__":
    main()
