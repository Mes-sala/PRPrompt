"""Collect every mixed-forward failure without changing any acceptance threshold.

The original frozen auditor remains authoritative and unchanged. This collector
defers its single mixed-forward assertion so the remaining units can be checked;
the deferred test's false status is explicitly stored, never converted to a pass.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path


class DeferMixedAssertion(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Assert(self, node):
        if ast.unparse(node.test) != "ratio < 1":
            return node
        self.changed += 1
        return ast.copy_location(ast.Expr(ast.Call(func=ast.Name("collect_mixed_assertion", ast.Load()),
            args=[ast.Name("path", ast.Load()), ast.Name("ratio", ast.Load()), ast.Name("diff", ast.Load())],
            keywords=[])), node)


def collect(args):
    source = args.source_root/"experiments/f100_multidata/audit.py"
    sys.path.insert(0, str(args.source_root)); sys.path.insert(0, str(args.source_root/"tools"))
    from prprompt.protocol import save_json, sha_file
    tree = ast.parse(source.read_text()); transform = DeferMixedAssertion()
    tree = transform.visit(tree); ast.fix_missing_locations(tree)
    assert transform.changed == 1, "Only the mixed-forward assertion may be deferred"
    failures = []

    def collect_mixed_assertion(path, ratio, diff):
        if ratio < 1:
            return
        entry = dict(unit=str(path.relative_to(args.plan.parent)), mixed_forward_error_ratio=ratio,
                     raw_forward_max_error=float(diff.max()), predeclared_mixed_forward_passed=False)
        failures.append(entry)
        print(json.dumps(entry), flush=True)

    namespace = dict(__name__="frozen_audit_collection", __file__=str(source),
                     collect_mixed_assertion=collect_mixed_assertion)
    exec(compile(tree, str(source), "exec"), namespace)

    def capture(path, value):
        assert path.name.endswith("_audit.json"), path
        value = dict(value, audit_mode="collection_with_unchanged_acceptance_test",
            deferred_assertions="ratio < 1 only; original source preserved; all other assertions enforced",
            original_auditor_sha256=sha_file(source), collector_sha256=sha_file(Path(__file__)),
            predeclared_mixed_forward_passed=not failures,
            predeclared_numerical_acceptance_passed=not failures,
            mixed_forward_failures=failures, mixed_forward_failure_count=len(failures),
            replay_device=args.device)
        output = args.plan.parent/"analysis/numerical_collection"/path.name
        save_json(output, value)
        print(json.dumps(dict(checkpoint_id=value["checkpoint_id"], present=value["present"],
            predeclared_numerical_acceptance_passed=not failures, mixed_forward_failures=len(failures))), flush=True)

    namespace["save_json"] = capture
    namespace["audit"](args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    collect(parser.parse_args())
