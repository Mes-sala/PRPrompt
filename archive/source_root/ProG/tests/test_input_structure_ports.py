"""Numerical contracts against the checked-out authors' code, plus task wiring."""

import ast
from pathlib import Path
from typing import Optional, Union

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import k_hop_subgraph, scatter, softmax

from prompt_graph.model import build_gnn
from prompt_graph.prompt.GraphTOP import (
    Rewiring,
    annealing_temperature,
    get_topology_subgraphs,
    prompted_edge_weights,
    sample_edges,
    topology_losses,
)
from prompt_graph.prompt.SUPT import DiffPoolPrompt, SAGPoolPrompt
from prompt_graph.tasker.input_structure import state_hash

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT.parent


def upstream_symbols(path, names):
    """Execute selected *unaltered* upstream definitions, bypass obsolete imports."""
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert len(nodes) == len(names)
    env = dict(
        torch=torch,
        nn=nn,
        F=torch.nn.functional,
        Tensor=Tensor,
        Optional=Optional,
        Union=Union,
        GCNConv=GCNConv,
        glorot=glorot,
        orthogonal=torch.nn.init.orthogonal_,
        scatter=scatter,
        softmax=softmax,
        Data=Data,
        k_hop_subgraph=k_hop_subgraph,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def small_graph():
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    return Data(x=torch.randn(7, 4), edge_index=edges, y=torch.tensor([0, 1, 0, 1, 0, 1, 0]))


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("branch", ["chem", "bio"])
@pytest.mark.parametrize("orth", [False, True])
def test_supt_upstream_forward_and_gradients(hard, branch, orth, monkeypatch):
    path = SOURCES / "SUPT" / branch / "graph_prompt.py"
    if not path.exists():
        pytest.skip("Author checkout unavailable; run this test with the recorded source revision")
    # Authors hard-code .cuda() for an identity constant. CPU-only test adaptation.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    env = upstream_symbols(path, {"SAGPoolPrompt", "DiffPoolPrompt", "topk"})
    name = "SAGPoolPrompt" if hard else "DiffPoolPrompt"
    kwargs = dict(in_channels=4, num_clusters=3, orth_loss=orth)
    if hard:
        kwargs["ratio"] = 0.5
    ref = env[name](**kwargs).double()
    port = (
        SAGPoolPrompt(**kwargs, official_branch=branch) if hard else DiffPoolPrompt(**kwargs)
    ).double()
    port.load_state_dict(ref.state_dict(), strict=False)  # port registers the constant eye buffer
    ref.eye = torch.eye(3, dtype=torch.double)
    data = small_graph()
    batch = Batch.from_data_list(
        [
            data,
            Data(
                x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([1])
            ),
        ]
    )
    x = batch.x.double()
    actual = port.add(x, batch.edge_index, batch.batch)
    expected = ref.add(x, batch.edge_index, batch.batch)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    weights = torch.randn_like(actual)
    loss_a, loss_b = (actual * weights).sum(), (expected * weights).sum()
    if orth:
        loss_a, loss_b = loss_a + port.orthogonal_loss(), loss_b + ref.orthogonal_loss()
    loss_a.backward()
    loss_b.backward()
    for a, b in zip(port.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)


def test_graphtop_subgraph_batching_matches_author_and_labels_are_not_inputs():
    graph = small_graph()
    path = SOURCES / "GraphTOP/load_data.py"
    if not path.exists():
        pytest.skip("Author checkout unavailable; run this test with the recorded source revision")
    ref = upstream_symbols(path, {"get_subgraphs"})["get_subgraphs"]
    expected = Batch.from_data_list(ref(graph, [0, 3]))
    actual = Batch.from_data_list(get_topology_subgraphs(graph, [0, 3]))
    for name in [
        "x",
        "edge_index",
        "edge_weight",
        "trainable_edge",
        "neighbor_mask",
        "target_node_index",
        "original_idx",
        "node_idx",
    ]:
        torch.testing.assert_close(actual[name], expected[name])
    changed = graph.clone()
    changed.y = 1 - changed.y
    other = Batch.from_data_list(get_topology_subgraphs(changed, [0, 3]))
    for name in ["x", "edge_index", "edge_weight", "node_idx"]:
        torch.testing.assert_close(actual[name], other[name])
    torch.testing.assert_close(actual.node_idx[actual.target_node_index], actual.original_idx)


def test_graphtop_sampling_and_regularizers_match_code():
    prob = torch.tensor([[0.1], [0.4], [0.9]], requires_grad=True)
    torch.manual_seed(31)
    u0, u1 = torch.rand_like(prob), torch.rand_like(prob)
    eps, tau = 1e-10, annealing_temperature(2, 10)
    noise = -torch.log(torch.log(u1 + eps) / torch.log(u0 + eps) + eps)
    expected = torch.sigmoid((torch.log(prob + eps) - torch.log(1 - prob + eps) + noise) / tau)
    torch.manual_seed(31)
    torch.testing.assert_close(sample_edges(prob, tau, True), expected)
    torch.manual_seed(32)
    expected = (prob > torch.rand_like(prob)).float()
    torch.manual_seed(32)
    torch.testing.assert_close(sample_edges(prob, tau, False), expected)
    batch = Batch.from_data_list(get_topology_subgraphs(small_graph(), [0, 3]))
    all_prob = torch.linspace(0.1, 0.9, batch.num_nodes).view(-1, 1).requires_grad_()
    ent, sparse = topology_losses(all_prob, batch)
    p = all_prob[batch.neighbor_mask]
    expected_ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()
    expected_sparse = torch.stack(
        [
            (
                all_prob[batch.ptr[i] : batch.ptr[i + 1]][
                    batch.neighbor_mask[batch.ptr[i] : batch.ptr[i + 1]]
                ].mean()
                - 0.5
            )
            ** 2
            for i in range(batch.num_graphs)
        ]
    ).mean()
    torch.testing.assert_close(ent, expected_ent)
    torch.testing.assert_close(sparse, expected_sparse)
    assert annealing_temperature(10, 10) == pytest.approx(0.03)


def test_graphtop_rewiring_gradient_frozen_encoder_and_isolated_node():
    batch = Batch.from_data_list(get_topology_subgraphs(small_graph(), [0, 3, 6]))
    gnn = build_gnn("GCN", 4, 8, 2).requires_grad_(False).eval()
    prompt = Rewiring(8)
    h = torch.randn(7, 8)
    before = state_hash(gnn)
    weights, prob = prompted_edge_weights(prompt, h, batch, 0.5, True)
    paired = weights[batch.trainable_edge].view(-1, 2)
    torch.testing.assert_close(paired[:, 0], paired[:, 1])
    out = gnn(batch.x, batch.edge_index, edge_weight=weights)[batch.target_node_index]
    ent, sparse = topology_losses(prob, batch)
    loss = out.square().sum() + ent + sparse
    loss.backward()
    assert sum(p.grad.abs().sum() for p in prompt.parameters()) > 0
    assert all(p.grad is None for p in gnn.parameters())
    assert state_hash(gnn) == before
    isolated = Batch.from_data_list(get_topology_subgraphs(small_graph(), [6]))
    ent, sparse = topology_losses(torch.tensor([[0.5]], requires_grad=True), isolated)
    assert ent.item() == sparse.item() == 0


@pytest.mark.parametrize("method", ["SUPT-soft", "SUPT-hard", "GraphTOP"])
def test_node_task_end_to_end_with_checkpoint(method, tmp_path, monkeypatch):
    import prompt_graph.tasker.input_structure as runner
    import prompt_graph.tasker.node_task as nt
    from prompt_graph.data.induced_graph import induced_graphs
    from prompt_graph.tasker import NodeTask

    graph = small_graph()
    graph.train_mask = torch.tensor([True, True, True, True, False, False, False])
    graph.test_mask = ~graph.train_mask
    monkeypatch.setattr(nt, "sample_dir", lambda *a: tmp_path / "splits")
    monkeypatch.setattr(runner, "sample_dir", lambda *a: tmp_path / "splits")
    checkpoint = tmp_path / "Cora" / "DGI.GCN.8hidden_dim.pth"
    checkpoint.parent.mkdir()
    torch.save(build_gnn("GCN", 4, 8, 2).state_dict(), checkpoint)
    task = NodeTask(
        data=graph,
        input_dim=4,
        output_dim=2,
        task_num=2,
        graphs_list=induced_graphs(graph, 1, 10),
        dataset_name="Cora",
        prompt_type=method,
        gnn_type="GCN",
        hid_dim=8,
        num_layer=2,
        pre_train_model_path=str(checkpoint),
        epochs=2,
        shot_num=1,
        batch_size=2,
        device="cpu",
        prompt_options={"supt_clusters": 2},
    )
    result = task.run()
    assert len(result) == 9 and 0 <= result[1] <= 1
    assert len(task.port_diagnostics) == 2
    assert (
        task.port_diagnostics[0]["encoder_hash_before"]
        == task.port_diagnostics[1]["encoder_hash_before"]
    )


@pytest.mark.parametrize("method", ["SUPT-soft", "SUPT-hard"])
def test_graph_task_split_training(method):
    from prompt_graph.tasker import GraphTask

    graphs = [
        Data(
            x=torch.randn(4, 4),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            y=torch.tensor([i % 2]),
        )
        for i in range(6)
    ]
    task = GraphTask(
        input_dim=4,
        output_dim=2,
        dataset=graphs,
        dataset_name="MUTAG",
        prompt_type=method,
        gnn_type="GCN",
        hid_dim=8,
        epochs=2,
        shot_num=0,
        device="cpu",
    )
    train, test = graphs[:4], graphs[4:]
    results = task._run_for_split(
        train, test, DataLoader(train, batch_size=2), DataLoader(test, batch_size=2)
    )
    assert 0 <= results[0] <= 1


def test_options_cli_and_unsupported_backbone():
    from prompt_graph.tasker.input_structure import input_structure_options
    from prompt_graph.utils.get_args import _build_parser

    args = _build_parser().parse_args(
        [
            "--supt_branch",
            "bio",
            "--supt_orth_loss",
            "--graphtop_lambda_e",
            "0.3",
            "--supt_ratio",
            "0.7",
        ]
    )
    options = input_structure_options(args)
    assert options["supt_branch"] == "bio" and options["supt_orth_loss"]
    assert options["graphtop_lambda_e"] == 0.3
    graph = small_graph()
    graph.edge_index = graph.edge_index[:, :1]
    with pytest.raises(ValueError, match="undirected"):
        get_topology_subgraphs(graph, [0])
