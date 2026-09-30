"""Run a fixed configuration and save auditable results, without editing old tables.

Example: python scripts/run_input_structure.py --output-dir results/ports/run1
         --config configs/supt_soft_cora.yaml
"""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    output, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    import torch
    import torch_geometric

    from downstream_task import get_downstream_task_delegate
    from prompt_graph.utils import apply_log_level, get_args

    args = get_args()
    if args.prompt_type not in {"SUPT-soft", "SUPT-hard", "GraphTOP"}:
        raise ValueError("This runner accepts SUPT-soft, SUPT-hard, GraphTOP")
    if args.downstream_task not in {"NodeTask", "GraphTask"}:
        raise ValueError("Use NodeTask or GraphTask")
    if args.prompt_type == "GraphTOP" and args.downstream_task != "NodeTask":
        raise ValueError("The official GraphTOP implementation only supports node classification")
    if args.downstream_task == "NodeTask":
        from run_protocol_cell import run

        from prompt_graph.tasker.input_structure import input_structure_options
        result = run(dict(
            dataset=args.dataset_name, prompt=args.prompt_type, checkpoint=args.pre_train_model_path,
            gnn=args.gnn_type, hidden_dim=args.hid_dim, layers=args.num_layer, shot=args.shot_num,
            folds=args.task_num, support_passes=args.epochs, learning_rate=args.lr,
            weight_decay=args.decay, split_seed=args.seed, train_seed=args.runseed,
            device=str(args.device), output_root=str(output.output_dir),
            split_root=str(ROOT / "Experiment/sample_data_v4"),
            prompt_options=input_structure_options(args)))
        print(json.dumps({"status": result["status"], "output_directory": result["output_directory"]}))
        return
    output.output_dir.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output.output_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))

    save("config.json", vars(args))
    tracked = [
        ROOT / "prompt_graph/prompt/SUPT.py",
        ROOT / "prompt_graph/prompt/GraphTOP.py",
        ROOT / "prompt_graph/tasker/strategies/input_structure.py",
        ROOT / "prompt_graph/tasker/input_structure.py",
        ROOT / "prompt_graph/model/gnn.py",
        ROOT / "prompt_graph/tasker/task.py",
        ROOT / "prompt_graph/tasker/graph_task.py",
    ]
    save(
        "implementation.json",
        {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "files": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tracked
            },
            "checkpoint_sha256": None
            if args.pre_train_model_path == "None"
            else hashlib.sha256(Path(args.pre_train_model_path).read_bytes()).hexdigest(),
            "encoder": "ProG backbone, original activation retained",
            "protocol": "ProG splits, fixed_last, no query checkpoint selection",
        },
    )
    apply_log_level(args.log_level, args.quiet)
    try:
        task = get_downstream_task_delegate(args)
        values = task.run()
        keys = [
            "support_objective",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "auroc_mean",
            "auroc_std",
            "auprc_mean",
            "auprc_std",
        ]
        if len(values) != len(keys):
            raise ValueError("Use positive shot_num for this few-shot report runner")
        save("metrics.json", dict(zip(keys, values)))
        save("diagnostics.json", getattr(task, "port_diagnostics", []))
    except Exception:
        save("failure.json", {"traceback": traceback.format_exc()})
        raise
    print(f"Results saved to {output.output_dir.resolve()}")


if __name__ == "__main__":
    main()
