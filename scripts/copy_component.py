"""Copia um componente para o build usando caminhos longos nativos do Windows."""
import argparse
import os
from pathlib import Path
import shutil


def extended(path: Path) -> str:
    value = str(path.resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def main():
    parser = argparse.ArgumentParser(description="Copia o componente validado para a árvore de build.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.target.resolve()
    if not target.is_relative_to(root / "build"):
        raise ValueError("O destino precisa estar dentro de build/ do projeto.")
    shutil.copytree(extended(args.source), extended(target), dirs_exist_ok=True)
    print(f"Componente copiado para {target}", flush=True)


if __name__ == "__main__":
    main()
