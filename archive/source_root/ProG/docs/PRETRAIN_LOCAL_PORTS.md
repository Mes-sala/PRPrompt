# Local pretraining ports

This integration reads the checked-in projects under the workspace-level
`../../pretrain/` directory and exposes
their encoder objectives through ProG's `pre_train.py` entry point.  All three
ports use `load4node`, ProG's `build_gnn`/`GNN` backbone, and save only the
encoder state dict at:

```text
Experiment/pre_trained_model/<dataset>/<method>.<gnn>.<hidden_dim>hidden_dim.pth
```

That is the format consumed by the existing downstream `pre_train_model_path`
loader.  Each method also accepts an injected PyG `Data` object, which keeps
the unit tests independent of dataset downloads.

## Integrated methods

| Method | Local source | Ported objective | Current scope |
| --- | --- | --- | --- |
| `GraphMAE2` | `../../pretrain/GraphMAE2` and GraphMAE2 paper | feature masking, decoder remasking, EMA latent target | full-batch node graphs |
| `S2GAE` | `../../pretrain/S2GAE` and S2GAE project | hide graph edges, reconstruct masked positives against sampled negatives | node graphs; `um`/`dm` masking |
| `Bandana` | `../../pretrain/Bandana` and Bandana paper | per-layer temperature-controlled continuous edge bandwidths plus link reconstruction | full-batch node graphs; ProG `GCN` |

The DGL-specific GraphMAE2 model and the custom sparse layers in Bandana were
not copied into ProG.  Their training signals were reimplemented on ProG's
PyG backbone so a checkpoint has the same parameter names and dimensions as a
normal ProG encoder.  Bandana is restricted to `GCN` because ProG's other
convolution operators do not share the `edge_weight` contract used by the
paper's encoder.

## Commands

```bash
python pre_train.py --config configs/graphmae2_cora.yaml
python pre_train.py --config configs/s2gae_cora.yaml
python pre_train.py --config configs/bandana_cora.yaml
```

The supplied Cora configurations are conservative starting points.  They are
not claims that one pretraining method is universally better; compare them
under the same seed, backbone, hidden size, and downstream protocol.

## Non-integrated local material

`../../ref/Pretrain` also contains papers on MGAE and structure-aware curriculum
masking.  There is no corresponding checked-in implementation in this
workspace, so they are recorded as candidates for a later paper-faithful port
rather than silently represented by one of the three implementations above.
`../../rep/DAPrompt` has a separate structure learner and a checkpoint format that
does not expose a ProG-compatible encoder; it remains a reference project
until its objective is evaluated under the same checkpoint contract.
