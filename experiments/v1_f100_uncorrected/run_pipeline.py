"""Run original-checkpoint F100 smoke, twelve disjoint two-GPU shards and sealed-state audit."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prprompt.protocol import save_json, sha_file
from experiments.v1_f100.run import read_plan


def main(args):
    plan = read_plan(args.plan)
    workspace = args.plan.parent
    started = time.time()

    def status(phase, **extra):
        save_json(workspace/"pipeline_status.json", {"phase": phase, "pid": os.getpid(),
                  "time": time.time(), "started": started, "plan_sha256": plan["plan_sha256"], **extra})

    def command_stage(command, log_path, phase):
        with log_path.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                status(phase, active_pid=process.pid)
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"{phase} failed; preserve results and inspect {log_path}")

    try:
        tests = workspace/"tests"
        tests.mkdir(exist_ok=True)
        for device in ["cpu", "cuda:0", "cuda:1"]:
            command_stage([sys.executable, "-m", "experiments.v1_f100.test_parity", "--device", device],
                          tests/f"{device.replace(':', '_')}.txt", "historical_parity")
        for smoke in [True, False]:
            phase = "smoke" if smoke else "fits"
            active, samples = [], []
            for index in range(len(plan["jobs"])):
                for shard in ([0] if smoke else range(plan["split_shards"])):
                    gpu = (index//2+index+shard) % 2
                    log = (workspace/f"{phase}_worker_{index}_{shard}.log").open("a")
                    command = [sys.executable, "-m", "experiments.v1_f100_uncorrected.run", "--plan", str(args.plan),
                               "--job-index", str(index), "--shard", str(shard), "--device", f"cuda:{gpu}"]
                    if smoke:
                        command.append("--smoke")
                    active.append((subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, index*3+shard, gpu))
            while any(p.poll() is None for p, _, _, _ in active):
                output = workspace/("smoke" if smoke else "results")
                completed = len(list(output.glob("*/5/*/*/F100/metrics.json")))
                status(phase, completed=completed, expected=4 if smoke else plan["expected_units"],
                       active=[{"pid": p.pid, "job_index": i, "gpu": gpu} for p, _, i, gpu in active if p.poll() is None])
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                               "--format=csv,noheader,nounits"], text=True)
                samples.append({"time": time.time(), "gpus": [list(map(int, row.split(','))) for row in raw.splitlines()]})
                if any(p.poll() not in [None, 0] for p, _, _, _ in active):
                    raise RuntimeError(f"A {phase} worker failed; preserve surviving processes and logs")
                time.sleep(2)
            for process, log, _, _ in active:
                log.close()
                assert process.returncode == 0
            save_json(workspace/f"{phase}_gpu_telemetry.json", samples)
            command = [sys.executable, "-m", "experiments.v1_f100_uncorrected.summarize", "--plan", str(args.plan)]
            if smoke:
                command.append("--smoke")
            command_stage(command, workspace/f"{phase}_audit.log", f"{phase}_audit")
        # Public truth is copied only after every fit and sealed-prediction audit completes.
        shutil.copytree(Path(plan["audit_reference_source"]), workspace/"audit_reference", dirs_exist_ok=True)
        manifest = {str(p.relative_to(workspace)): sha_file(p) for p in workspace.rglob('*')
                    if p.is_file() and not p.is_symlink() and '__pycache__' not in str(p)
                    and p.name not in ['pipeline_status.json', 'result_files_sha256.json']}
        save_json(workspace/"result_files_sha256.json", manifest)
        status("complete", finished=time.time(), completed=plan["expected_units"], manifest_files=len(manifest),
               report="analysis/未修正Edgepred_F100验证总结.md")
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    main(parser.parse_args())
