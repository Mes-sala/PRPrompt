"""Results-only local storage policy; scientific results and NPZ diagnostics stay local."""
from pathlib import PurePosixPath

POLICY = 'code-results-documents-no-model-binaries-v1'
REMOTE_ONLY_SUFFIXES = ('.pt', '.pt.gz', '.pth', '.pth.gz', '.ckpt', '.ckpt.gz',
                        '.safetensors', '.onnx', '.h5', '.hdf5', '.bin', '.pkl', '.pickle')


def remote_only(name):
    path = PurePosixPath(str(name))
    # Keep checkpoint sidecars, provenance and all result/diagnostic arrays.
    return path.name.lower().endswith(REMOTE_ONLY_SUFFIXES)


def rsync_exclusions():
    return ['--exclude=*' + suffix for suffix in REMOTE_ONLY_SUFFIXES] + [
        '--exclude=__pycache__', '--exclude=*.tmp', '--exclude=*.lock',
        '--exclude=.DS_Store', '--exclude=._*', '--exclude=__MACOSX/',
        '--exclude=/data/', '--exclude=*.tar', '--exclude=*.tar.gz', '--exclude=*.tar.zst', '--exclude=*.tgz']


def verify_remote_receipt(receipt, hashes, workspace, manifest_sha256):
    assert receipt['schema'] == POLICY
    assert receipt['remote_bytes_sha256_verified'] is True
    assert receipt['result_manifest_sha256'] == manifest_sha256
    excluded = {name: expected for name, expected in hashes.items() if remote_only(name)}
    assert set(receipt['files']) == set(excluded), 'Remote-only receipt coverage mismatch'
    for name, expected in excluded.items():
        assert receipt['files'][name]['sha256'] == expected
        assert not (workspace / name).exists(), 'Model binary must not remain locally: ' + name
    return excluded
