"""Read-only remote SHA256 retention check for binaries excluded from local delivery.

Run in the remote workspace after terminal completion, then copy the JSON receipt
locally. This changes neither the sealed scientific source nor experiment outputs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from local_artifact_policy import POLICY, remote_only


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root, hashes, manifest_sha256):
    files = {}
    for name, expected in sorted(hashes.items()):
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        if not remote_only(name):
            continue
        path = root / relative
        assert path.is_file() and sha(path) == expected, name
        files[name] = dict(sha256=expected, bytes=path.stat().st_size,
                           remote_path=str(path.resolve()))
        if len(files) % 1000 == 0:
            print(json.dumps(dict(remote_model_files_verified=len(files))), flush=True)
    return dict(schema=POLICY, remote_workspace=str(root.resolve()),
                result_manifest_sha256=manifest_sha256,
                remote_bytes_sha256_verified=True, verified_at=time.time(), files=files)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-hashes', type=Path)
    args = p.parse_args()
    root = args.workspace.resolve()
    assert json.loads((root / 'pipeline_status.json').read_text())['phase'] == 'server_complete'
    manifest = root / 'result_files_sha256.json'
    hashes = json.loads(manifest.read_text())
    if args.extra_hashes:
        extra = json.loads(args.extra_hashes.read_text())
        for name in set(extra) & set(hashes):
            assert extra[name] == hashes[name]
        hashes.update(extra)
    receipt = verify(root, hashes, sha(manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(args.output)
