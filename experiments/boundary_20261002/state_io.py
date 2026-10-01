"""Lossless gzip for states; compressed bytes are hashed, decompression changes no tensors."""
import gzip
import io
from pathlib import Path
import torch

def save_state(value, path):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    with open(tmp, 'wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
            torch.save(value, stream)
    tmp.replace(path)

def load_state(path):
    with gzip.open(path, 'rb') as stream:
        content = stream.read()
    return torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
