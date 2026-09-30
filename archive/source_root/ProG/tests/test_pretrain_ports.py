from pathlib import Path

import torch
from torch_geometric.data import Data

from prompt_graph.defines import PRETRAIN_TYPES
from prompt_graph.model import build_gnn
from prompt_graph.pretrain import S2GAE, Bandana, GraphMAE2
from prompt_graph.utils.get_args import get_args_by_call


def toy_graph():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 2, 4, 1, 3], [1, 2, 3, 4, 0, 2, 4, 1, 3, 0]],
        dtype=torch.long,
    )
    return Data(x=torch.randn(5, 4), edge_index=edge_index)


def test_new_pretrains_export_pro_gnn_checkpoints(tmp_path: Path):
    data = toy_graph()
    methods = [
        (GraphMAE2, {}),
        (S2GAE, {"mask_type": "um"}),
        (Bandana, {}),
    ]
    for method, options in methods:
        target = tmp_path / f"{method.__name__}.pth"
        model = method(
            data=data,
            dataset_name="synthetic",
            gnn_type="GCN",
            hid_dim=8,
            gln=2,
            num_epoch=1,
            device="cpu",
            output_path=target,
            **options,
        )
        loss = model.pretrain_one_epoch()
        assert loss >= 0
        checkpoint = model.pretrain()
        assert checkpoint == str(target)
        loaded = build_gnn("GCN", 4, 8, 2)
        loaded.load_state_dict(torch.load(target, weights_only=True))
        assert loaded(data.x, data.edge_index).shape == (5, 8)


def test_pretrain_registry_and_defaults_include_local_ports():
    assert {"GraphMAE2", "S2GAE", "Bandana"}.issubset(PRETRAIN_TYPES)
    args = get_args_by_call()
    assert args.graphmae2_mask_rate == 0.5
    assert args.s2gae_mask_type == "dm"
    assert args.bandana_temp == (1.0,)


def test_s2gae_masking_does_not_reintroduce_masked_edges():
    from torch_geometric.utils import to_undirected
    data = toy_graph()
    data.edge_index = to_undirected(data.edge_index)
    for kind in ('um', 'dm'):
        model = S2GAE(data=data, dataset_name='synthetic', gnn_type='GCN',
                      hid_dim=8, gln=2, num_epoch=1, device='cpu', mask_type=kind)
        for _ in range(8):
            visible, masked = model._mask_edges()
            seen = set(map(tuple, visible.t().tolist()))
            hidden = set(map(tuple, masked.t().tolist()))
            assert not seen & hidden
            if kind == 'um':
                assert not seen & {(b, a) for a, b in hidden}


def test_masked_checkpoint_has_verifiable_provenance(tmp_path):
    import hashlib
    import json
    model = S2GAE(data=toy_graph(), dataset_name='synthetic', gnn_type='GCN',
                  hid_dim=8, gln=2, num_epoch=1, device='cpu', output_path=tmp_path/'model.pth')
    path = Path(model.pretrain())
    metadata = json.loads(Path(str(path)+'.json').read_text())
    assert metadata['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata['completed_epochs'] == 1
    assert metadata['labels_used'] is False
    assert metadata['variant'] == 'ProG-adapted'
