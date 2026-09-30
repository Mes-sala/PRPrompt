"""Smoke and unit tests for DHBP."""

import argparse

import pytest
import torch

bench = pytest.importorskip("bench")


def _ns(**overrides):
    base = dict(
        pretrain_task="NodeTask",
        dataset_name="Cora",
        prompt_type="DHBP",
        gnn_type="GCN",
        num_layer=2,
        hid_dim=128,
        epochs=1,
        shot_num=5,
        device="cpu",
        pre_train_model_path="None",
        batch_size=64,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_strategy_registry_has_dhbp():
    from prompt_graph.tasker.strategy import STRATEGY_REGISTRY

    assert "DHBP" in STRATEGY_REGISTRY


def test_dhbp_prompt_is_identity_at_initial_scale():
    from prompt_graph.prompt.DHBP import DHBPPrompt

    prompt = DHBPPrompt(in_channels=4, num_classes=2, num_iters=2)
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2], [1, 2, 3, 4, 0, 1]],
        dtype=torch.long,
    )
    boundary_idx = torch.tensor([0, 3], dtype=torch.long)
    boundary_y = torch.tensor([0, 1], dtype=torch.long)

    out = prompt.add(x, edge_index, boundary_idx, boundary_y)

    assert torch.allclose(out, x)
    assert prompt.last_diagnostics["dhbp_intervention_ratio"] == 0.0


def test_node_task_cora_dhbp_strategy():
    args = _ns()
    result = bench.do_config_bench(args)
    assert result.final_acc_mean is not None
    assert 0.0 <= result.final_acc_mean <= 1.0
