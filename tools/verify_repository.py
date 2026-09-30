"""Verify consolidated source bytes and the unchanged PRPrompt runtime; no ML dependencies."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    count = 0
    for line in (ROOT / 'provenance/files.sha256.jsonl').read_text().splitlines():
        row = json.loads(line)
        path = ROOT / row['path']
        require(path.is_file(), f'Missing source: {row["path"]}')
        require(sha(path) == row['sha256'], f'Changed source: {row["path"]}')
        count += 1
    runtime = ROOT / 'runtime'
    base = runtime / 'base'
    manifest = base / 'source_manifest.json'
    sources = json.loads(manifest.read_text())
    for row in sources:
        require(sha(base / row['path']) == row['sha256'], f'Frozen dependency changed: {row["path"]}')
    require(sha(manifest) == json.loads((base / 'plan.json').read_text())['source_manifest_sha256'], 'Frozen manifest changed')
    require(sha(runtime / 'run_softening.py') == json.loads((runtime / 'plan.json').read_text())['runner_sha256'], 'Frozen runner changed')
    scripts = list(runtime.rglob('*.py'))
    for path in scripts:
        ast.parse(path.read_bytes(), filename=str(path))
    print(f'PASS: {count} source files, {len(sources)} frozen dependencies, {len(scripts)} Python syntax checks')
    print('No training was run. Checkpoints and datasets must be restored separately.')


if __name__ == '__main__':
    main()
