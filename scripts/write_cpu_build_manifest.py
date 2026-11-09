"""Registra o pacote local CPU e seu ambiente sem publicar artefatos."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Registra hashes e dependências do pacote CPU.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    files = sorted(path for path in package.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Pacote vazio ou ausente: {package}")
    entries = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"path": path.relative_to(package).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    result = {
        "status": "PRELIMINARY_LOCAL_CPU_BUILD", "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision, "working_tree_dirty": bool(dirty), "python": sys.version,
        "dependencies": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "package": str(package), "file_count": len(entries), "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        "limitations": ["Validação em outro computador pendente", "Revisão de licenças e direitos para redistribuição pendente"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifesto local gravado: {args.output} ({len(entries)} arquivos).")


if __name__ == "__main__":
    main()
