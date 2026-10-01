"""Two-GPU memory-budgeted execution, complete audit and immutable delivery bundle."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from prprompt.protocol import save_json, sha_file
from experiments.f100_multidata.run import read_plan


def pipeline(workspace):
    plan = read_plan(workspace/"plan.json"); planfile = workspace/"plan.json"
    statusfile = workspace/"pipeline_status.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    logroot = workspace/"logs"; logroot.mkdir(exist_ok=True)
    status = dict(pid=os.getpid(), phase="parity", plan_sha256=plan["plan_sha256"], started_at=time.time(), completed_tasks=[])

    def update(**fields):
        status.update(fields, updated_at=time.time()); save_json(statusfile, status)

    def command(module, args, logfile):
        with logfile.open("x") as stream:
            result = subprocess.run([sys.executable, "-m", module]+args, cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, str(logfile)

    def dispatch(tasks, phase, budgets=None):
        update(phase=phase, active=[], pending=len(tasks)); active = []; pending = list(tasks); failed = []
        while active or pending:
            if not failed:
                for gpu in [0, 1]:
                    while sum(t["gpu"] == gpu for t in active) < (6 if budgets else 1):
                        used = sum(t["budget_bytes"] for t in active if t["gpu"] == gpu)
                        index = next((i for i, task in enumerate(pending)
                            if used+(budgets[task["job"]] if budgets else 0) <= 28*1024**3), None)
                        if index is None:
                            break
                        task = pending.pop(index); module = task["module"]
                        arguments = ["--plan", str(planfile), "--job-index", str(task["job"]), "--device", f"cuda:{gpu}"]
                        if "shard" in task:
                            arguments += ["--shard", str(task["shard"])]
                        if phase.startswith("smoke"):
                            arguments += ["--smoke"]
                        logpath = logroot/(phase+"_"+str(task["job"])+"_"+str(task.get("shard", 0))+".log")
                        stream = logpath.open("x")
                        process = subprocess.Popen([sys.executable, "-m", module]+arguments,
                            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        active.append(dict(task, gpu=gpu, process=process, stream=stream, pid=process.pid,
                            logfile=str(logpath), budget_bytes=budgets[task["job"]] if budgets else 0))
            finished = []
            for task in active:
                code = task["process"].poll()
                if code is None:
                    continue
                task["stream"].close(); finished.append(task)
                record = {k: v for k, v in task.items() if k not in ["process", "stream"]}
                record.update(exit_code=code, finished_at=time.time())
                if code:
                    failed.append(record)
                else:
                    status["completed_tasks"].append(record)
            active = [task for task in active if task not in finished]
            public = [{k: v for k, v in t.items() if k not in ["process", "stream"]} for t in active]
            update(active=public, pending=len(pending), failed=failed)
            if failed and not active:
                raise RuntimeError("Stage failed; successful results and all logs preserved: "+str(failed))
            if pending and not active and not failed and budgets and min(budgets[t["job"]] for t in pending) > 28*1024**3:
                raise RuntimeError("A task exceeds the predeclared memory budget; review smoke, do not discard results")
            if active or pending:
                time.sleep(5)

    try:
        for device in ["cpu", "cuda:0", "cuda:1"]:
            command("experiments.v1_f100.test_parity", ["--device", device], logroot/("parity_"+device.replace(":", "_")+".log"))
        dispatch([dict(job=i, shard=0, module="experiments.f100_multidata.run") for i in range(len(plan["jobs"]))], "smoke_fit")
        dispatch([dict(job=i, module="experiments.f100_multidata.audit") for i in range(len(plan["jobs"]))], "smoke_audit")
        smoke_rows = [json.loads(p.read_text()) for p in (workspace/"smoke").glob("*/5/*/*/*/metrics.json")]
        assert len(smoke_rows) == len(plan["jobs"])*len(plan["methods"]) == 90
        budgets = {}; calibration = []
        for i, job in enumerate(plan["jobs"]):
            rows = [r for r in smoke_rows if r["checkpoint_id"] == job["id"]]
            peak = max(max(r["peak_cuda_bytes"], r["peak_reserved_cuda_bytes"]) for r in rows)
            budget = int(1.3*peak+1.5*1024**3); budgets[i] = budget
            calibration.append(dict(job_index=i, checkpoint_id=job["id"], smoke_peak_bytes=peak, budget_bytes=budget,
                per_gpu_budget_bytes=28*1024**3, maximum_workers_per_gpu=6,
                smoke_total_seconds=sum(r["total_seconds"] for r in rows)))
        save_json(workspace/"memory_calibration.json", calibration)
        order = sorted(range(len(plan["jobs"])), key=lambda i: -budgets[i])
        dispatch([dict(job=i, shard=s, module="experiments.f100_multidata.run") for i in order for s in range(3)], "formal_fit", budgets)
        dispatch([dict(job=i, module="experiments.f100_multidata.audit") for i in range(len(plan["jobs"]))], "formal_audit")
        update(phase="summarize")
        command("experiments.f100_multidata.summarize", [str(workspace)], logroot/"summarize.log")
        # Public labels for local independent scoring, exported after formal prediction sealing.
        provenance = []
        for dataset, spec in plan["datasets"].items():
            data = torch.load(spec["snapshot"], map_location="cpu", weights_only=True)
            target = workspace/"audit_reference"/(dataset+".npz"); target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, labels=data["labels"].numpy(), pool=data["pool"].numpy(), query=data["query"].numpy())
            provenance.append(dict(dataset=dataset, file=target.name, sha256=sha_file(target),
                graph_sha256=spec["graph_sha256"], labels_sha256=spec["labels_sha256"], exported_after_sealing=True))
        save_json(workspace/"audit_reference/provenance.json", provenance)
        with (workspace/"environment.txt").open("w") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
        # No more writes to logs or pipeline_status after this transition.
        update(phase="server_complete", active=[], pending=0, finished_at=time.time(),
               formal_units=plan["expected_units"], next_action="Local rsync, SHA + NumPy audit, Chinese report and Git archival")
        # Detached launcher stdout is diagnostic; copy its settled bytes into the sealed logs.
        launchlog = workspace/"pipeline.log"
        if launchlog.exists():
            (logroot/"controller.log").write_bytes(launchlog.read_bytes())
        manifest = {}
        for path in sorted(workspace.rglob("*")):
            relative = path.relative_to(workspace)
            if not path.is_file() or path.is_symlink() or path.name == "result_files_sha256.json":
                continue
            if relative.parts[0] == "data" or "__pycache__" in relative.parts or path.name in ["pipeline.log", "pipeline.pid"]:
                continue
            if path.suffix == ".lock":
                raise RuntimeError("Unexpected remaining running lock: "+str(path))
            manifest[str(relative)] = sha_file(path)
        save_json(workspace/"result_files_sha256.json", manifest)
    except Exception as error:
        update(phase="failed", error=repr(error)); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("workspace", type=Path)
    pipeline(parser.parse_args().workspace)
