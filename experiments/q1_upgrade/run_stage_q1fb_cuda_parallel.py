"""
Stage Q1-FB: Parallel CUDA Orchestrator for Full-Budget Hopper Confirmation.
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
    parser = argparse.ArgumentParser(description="Parallel CUDA Orchestrator for Stage Q1-FB MBPO.")
    parser.add_argument("--gpus", type=int, nargs="+", default=[3], help="GPU device IDs to use (e.g. 3 or 0 1 2 3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[142, 143, 144, 145, 146, 147, 148, 149, 150, 151], help="Confirmatory seed cohort")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["mle", "vagram"], help="Algorithms to evaluate")
    parser.add_argument("--total-steps", type=int, default=125000, help="Total environment interaction steps per seed")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of deterministic evaluation episodes per checkpoint")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Maximum concurrent worker processes")
    parser.add_argument("--out-dir", type=str, default=os.path.join(_ROOT, "results", "stage_q1fb_cuda"), help="Output directory for results and logs")
    return parser.parse_args()


def run_single_job(job_args: Tuple[str, int, int, int, int, int, str, str]) -> Dict[str, Any]:
    algo, seed, total_steps, eval_freq, eval_episodes, gpu_id, out_dir, python_bin = job_args
    
    out_file = os.path.join(out_dir, f"q1fb_hopper_{algo}_seed{seed}.json")
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
                "wall_clock": res.get("total_wall_clock_sec", 0.0),
                "out_file": out_file,
            }
        except Exception:
            pass  # Re-run if file corrupted

    cmd = [
        python_bin,
        "-u",
        os.path.join(_ROOT, "experiments", "q1_upgrade", "run_stage_q1fb_full_budget_hopper.py"),
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
        log_f.write(f"=== Stage Q1-FB Run: {algo.upper()} Seed {seed} on physical GPU {gpu_id} (isolated cuda:0) ===\n")
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
        term_ret = res.get("terminal_return_mean", 0.0)
    except Exception:
        term_ret = 0.0
        
    return {
        "algorithm": algo,
        "seed": seed,
        "gpu": gpu_id,
        "status": "completed",
        "terminal_return": term_ret,
        "wall_clock": wall_clock,
        "out_file": out_file,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    python_bin = sys.executable

    print("=" * 80)
    print("STAGE Q1-FB: PARALLEL CUDA ORCHESTRATOR (125k STEPS CONFIRMATORY COHORT)")
    print("=" * 80)
    print(f"Target GPUs:        {args.gpus}")
    print(f"Max Concurrency:    {args.max_concurrent} simultaneous workers")
    print(f"Algorithms:         {args.algorithms}")
    print(f"Confirmatory Seeds: {args.seeds} (N={len(args.seeds)})")
    print(f"Total Steps/Seed:   {args.total_steps}")
    print(f"Output Directory:   {args.out_dir}")
    print("=" * 80)

    # Build queue: interleaved pairs (seed MLE, seed VaGraM)
    jobs = []
    gpu_cycle = 0
    for seed in args.seeds:
        for algo in args.algorithms:
            gpu_id = args.gpus[gpu_cycle % len(args.gpus)]
            gpu_cycle += 1
            jobs.append((algo, seed, args.total_steps, args.eval_freq, args.eval_episodes, gpu_id, args.out_dir, python_bin))

    total_jobs = len(jobs)
    print(f"\nPrepared {total_jobs} total runs across {len(args.seeds)} paired seed blocks.")
    print(f"Worker pool size: {args.max_concurrent} processes.")

    start_time = time.time()
    completed_count = 0

    with ProcessPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        
        for future in as_completed(futures):
            job = futures[future]
            algo, seed, *_ = job
            completed_count += 1
            try:
                res = future.result()
                status = res["status"]
                elapsed_total = time.time() - start_time
                if status == "cached":
                    print(f"[{completed_count:02d}/{total_jobs:02d}] [CACHE] {algo.upper():6s} Seed {seed} | TermRet: {res['terminal_return']:7.2f} (took {res['wall_clock']:.1f}s)")
                elif status == "completed":
                    print(f"[{completed_count:02d}/{total_jobs:02d}] [DONE]  {algo.upper():6s} Seed {seed} on GPU {res['gpu']} | TermRet: {res['terminal_return']:7.2f} | Wall: {res['wall_clock']:.1f}s | TotalElapsed: {elapsed_total/60:.1f}m")
                else:
                    print(f"[{completed_count:02d}/{total_jobs:02d}] [FAIL]  {algo.upper():6s} Seed {seed} on GPU {res['gpu']} | ReturnCode: {res.get('returncode')} | Check log: {res.get('log_file')}")
            except Exception as exc:
                print(f"[{completed_count:02d}/{total_jobs:02d}] [EXC]   {algo.upper():6s} Seed {seed} generated exception: {exc}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL {total_jobs} RUNS FINISHED IN {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print("=" * 80)

    # Automatically trigger blind evaluation
    print("\n---> Triggering Automated Blind Statistical Evaluation <---")
    eval_cmd = [
        python_bin,
        os.path.join(_ROOT, "experiments", "q1_upgrade", "evaluate_stage_q1fb.py"),
        "--results-dir", args.out_dir,
        "--report-out", os.path.join(_ROOT, "docs", "STAGE_Q1FB_FULL_BUDGET_REPORT_CUDA.md"),
        "--fig-out", os.path.join(_ROOT, "figures", "stage_q1fb_full_budget_curves_cuda.pdf"),
    ]
    subprocess.run(eval_cmd, cwd=_ROOT)
    print("\nStage Q1-FB parallel execution and evaluation pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
