ProG environment prepared on 2026-09-16.

Activate:
  source /root/autodl-tmp/prog_env/bin/activate

Environment path:
  /root/autodl-tmp/prog_env

After ProG code transfer finishes, install project code from the ProG root, for example:
  cd /path/to/ProG
  python -m pip install -e '.[dev]'

Data/checkpoint recommendation:
  keep large datasets/checkpoints under /root/autodl-tmp, not /, because / is 30G and /root/autodl-tmp is 50G.

Installed and smoke-tested:
  torch 2.12.1+cu130 from base image via --system-site-packages
  torch_geometric, pyg_lib, torch_scatter, torch_sparse
  numpy<2, scipy, pandas, openpyxl, scikit-learn, torchmetrics, ogb, networkx, deprecated, tqdm, pyyaml, pytest

Known boundary:
  torch_cluster is not available from the official PyG torch-2.12.1+cu130 wheel page. A source build was attempted with --no-build-isolation but did not finish in a reasonable time and was interrupted. ProG paths that directly import torch_cluster.random_walk may need either a compatible wheel/source build or a small code compatibility change.

Freeze:
  /root/autodl-tmp/prog_env_freeze.txt

Project install: /root/ProG editable installed.
Pytest: 95 passed, 9 skipped, 2 failed before manual interrupt (GraphPrompter/PSP LinkTask loss lacks grad_fn), 1 deselected.
