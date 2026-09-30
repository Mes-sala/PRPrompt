#!/usr/bin/env python
"""Hold ~25 GB of VRAM on GPU0 for a fixed duration (default 1 hour).

Used to reserve GPU0 so the scheduler / other jobs leave it alone.
Allocates the memory as float32 tensors and keeps references alive, then
sleeps.  Duration (seconds) is read from argv[1] if given.
"""
import sys
import time

import torch

DEVICE = "cuda:0"
TARGET_BYTES = 25000 * 1024 * 1024  # 25,000 MiB
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0

if not torch.cuda.is_available():
    raise SystemExit("no CUDA available")

total = torch.cuda.get_device_properties(0).total_memory
print(
    f"GPU0 total memory: {total / 1024**2:.0f} MiB, "
    f"target hold: {TARGET_BYTES / 1024**2:.0f} MiB for {DURATION:.0f}s",
    flush=True,
)

chunks = []
allocated = 0
chunk_bytes = 1024 * 1024 * 1024  # 1 GiB per tensor
while allocated < TARGET_BYTES:
    n = chunk_bytes // 4  # float32 elements
    t = torch.empty(n, dtype=torch.float32, device=DEVICE)
    t.fill_(1.0)  # touch so it is actually committed
    chunks.append(t)
    allocated += chunk_bytes
    print(f"allocated {allocated / 1024**2:.0f} MiB", flush=True)

torch.cuda.synchronize()
used = torch.cuda.memory_allocated(0)
print(f"held: {used / 1024**2:.0f} MiB; sleeping {DURATION:.0f}s", flush=True)

# Keep references alive; just sleep.
time.sleep(DURATION)
print("done", flush=True)
