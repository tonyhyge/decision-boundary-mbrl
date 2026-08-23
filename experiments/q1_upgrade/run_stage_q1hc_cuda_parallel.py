"""
Stage Q1-HC: Parallel CUDA Orchestrator for HalfCheetah-v4 Confirmatory Cohort.
Coordinates concurrent execution of 20 runs (N=10 paired seeds, MLE vs VaGraM) across designated GPUs.
"""
import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def execute_single_job(job_args):
    algo, seed, total_steps, eval_freq, eval_episodes, gpu_id, out_dir, python_bin = job_args
    
    out_file = os.path.join(out_dir, f"q1hc_halfcheetah_{algo}_seed{seed}.json")
    log_file = os.path.join(out_dir, f"log_{algo}_seed{seed}.log")
    
    if os.path.exists(out_file):
        return f"[CACHED] {algo.upper()} Seed {seed} already exists."
        
    cmd = [
        python_bin, "-u",
        os.path.join(_ROOT, "experiments", "q1_upgrade", "run_stage_q1hc_halfcheetah.py"),
        "--algorithms", algo,
        "--seeds", str(seed),
        "--total-steps", str(total_steps),
        "--eval-freq", str(eval_freq),
        "--eval-episodes", str(eval_episodes),
        "--device", f"cuda:0",  # isolated device within worker process
        "--out-dir", out_dir,
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    start_t = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        
    dur_min = (time.time() - start_t) / 60.0
    if proc.returncode == 0:
        return f"[COMPLETED] {algo.upper()} Seed {seed} on GPU {gpu_id} | Wall: {dur_min:.1f} min"
    else:
        return f"[FAILED] {algo.upper()} Seed {seed} on GPU {gpu_id} with code {proc.returncode} (check {log_file})"


def main():
    parser = argparse.ArgumentParser(description="Parallel HalfCheetah Confirmatory Runner")
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 2, 3, 3], help="GPU IDs to allocate workers to")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(142, 152)))
    parser.add_argument("--algorithms", nargs="+", default=["mle", "vagram"])
    parser.add_argument("--total-steps", type=int, default=125000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="results/stage_q1hc_cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    python_bin = sys.executable

    print("=" * 80)
    print("STAGE Q1-HC: PARALLEL CUDA ORCHESTRATOR FOR HALFCHEETAH-V4 REPLICATION")
    print("=" * 80)
    print(f"Target Environment:    HalfCheetah-v4")
    print(f"Total Confirmatory N:  {len(args.seeds)} paired seeds ({args.seeds})")
    print(f"Algorithms:            {args.algorithms}")
    print(f"Budget per Run:        {args.total_steps} interaction steps")
    print(f"Designated GPUs:       {args.gpus}")
    print(f"Max Concurrent:        {args.max_concurrent} workers")
    print(f"Output Directory:      {args.out_dir}")
    print("=" * 80)

    jobs = []
    gpu_pool = args.gpus
    for i, seed in enumerate(args.seeds):
        for algo in args.algorithms:
            gpu_assigned = gpu_pool[(len(jobs)) % len(gpu_pool)]
            jobs.append((algo, seed, args.total_steps, args.eval_freq, args.eval_episodes, gpu_assigned, args.out_dir, python_bin))

    print(f"Total jobs queued: {len(jobs)} runs ({len(args.seeds)} paired seeds x {len(args.algorithms)} algorithms)")
    start_all = time.time()

    completed_count = 0
    with ProcessPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {executor.submit(execute_single_job, j): j for j in jobs}
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            print(f"{res} ({completed_count}/{len(jobs)} finished)")

    total_h = (time.time() - start_all) / 3600.0
    print("\n" + "=" * 80)
    print(f"ALL Q1-HC RUNS PROCESSED in {total_h:.2f} hours")
    print("=" * 80)


if __name__ == "__main__":
    main()
