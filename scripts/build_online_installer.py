"""Gera o instalador CPU com downloads GPU oficiais e impõe a meta de 500 MB."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.gpu_packages import load_catalog


def quote(value):
    return '"'+str(value).replace('"', '""')+'"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'dist/cpu/win-unpacked')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/online-installer')
    args = parser.parse_args()
    version = json.loads((ROOT/'desktop/package.json').read_text())['version']
    source = args.source.resolve()
    component = json.loads((source/'resources/backend/component.json').read_text(encoding='utf-8-sig'))
    if component['mode'] != 'cpu' or component['app_version'] != version:
        raise RuntimeError('Reconstrua primeiro o pacote CPU da versão atual.')
    catalog_path = ROOT/'packaging/gpu_catalog.json'
    catalog = load_catalog(catalog_path)
    work = ROOT/'build/online-installer'; work.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    entries = []
    for mode, packages in catalog['modes'].items():
        entries.append(f"if SelectedMode = '{mode}' then begin")
        for package in packages:
            values = [package[key].replace("'", "''") for key in ('url', 'filename', 'sha256')]
            entries.append("  DownloadPage.Add('"+"', '".join(values)+"');")
        entries.append('end;')
    downloads = work/'downloads.iss'
    downloads.write_text('\n'.join(entries)+'\n', encoding='utf-8-sig')
    defines = {'AppVersion': version, 'BaseSource': source, 'CatalogSource': catalog_path,
               'OutputFolder': output, 'DownloadEntries': downloads,
               'NvidiaDownloadMB': str(round(sum(p['bytes'] for p in catalog['modes']['nvidia'])/1e6)),
               'AmdDownloadMB': str(round(sum(p['bytes'] for p in catalog['modes']['amd'])/1e6))}
    recipe = work/'setup.iss'
    recipe.write_text('\n'.join(f'#define {key} {quote(value)}' for key, value in defines.items())+
                      '\n#include '+quote(ROOT/'packaging/windows_online_installer.iss')+'\n', encoding='utf-8-sig')
    subprocess.run([str(ROOT/'build/tools/inno7/ISCC.exe'), str(recipe)], check=True)
    installer = output/f'GoalEventReview-{version}-Online-Setup-x64.exe'
    size = installer.stat().st_size
    if size >= 500_000_000:
        raise RuntimeError(f'A meta de tamanho não foi atingida: {size} bytes. Não distribuir como pacote abaixo de 500 MB.')
    with installer.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'version': version, 'bytes': size, 'sha256': digest, 'cpu_offline': True,
                'gpu_downloads_bytes': {mode: sum(p['bytes'] for p in packages) for mode, packages in catalog['modes'].items()}}
    installer.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (output/'SHA256SUMS.txt').write_text(digest+'  '+installer.name+'\n', encoding='utf-8')
    (output/'LEIA-ME-TESTE.txt').write_text((ROOT/'docs/guia_teste_instalador_online.txt').read_text(encoding='utf-8'), encoding='utf-8-sig')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
