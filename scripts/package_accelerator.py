"""Cria um ZIP64 do componente isolado, sem vídeos ou dados de avaliação."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Empacota o componente GPU isolado.")
    parser.add_argument("--mode", choices=("nvidia", "amd"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.source / "goal-review-api.exe").is_file():
        raise FileNotFoundError("O executável do componente não foi encontrado.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(args.source.rglob("*")):
            if path.is_file():
                total += path.stat().st_size
                archive.write(path, (Path(args.mode) / path.relative_to(args.source)).as_posix())
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {"mode": args.mode, "archive": str(args.output), "download_bytes": args.output.stat().st_size,
              "installed_bytes": total, "sha256": digest.hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
