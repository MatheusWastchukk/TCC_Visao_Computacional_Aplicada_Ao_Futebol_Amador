"""Cria o executável completo offline com LZMA2 e memória limitada."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-engine-check', action='store_true', help='Somente para teste com aplicativo artificial.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    normal_source = args.source.resolve()
    source = Path('\\\\?\\' + str(normal_source)) if os.name == 'nt' else normal_source
    if not (source / 'Goal Event Review.exe').is_file():
        raise FileNotFoundError('A pasta deve conter Goal Event Review.exe.')
    if not args.skip_engine_check:
        for relative in ('backend', 'engines/nvidia', 'engines/amd'):
            if not (source / 'resources' / relative / 'goal-review-api.exe').is_file():
                raise FileNotFoundError(f'Componente ausente: {relative}')
    tools = root / 'build/tools/7zip'
    if not (tools/'7zr.exe').is_file():
        raise FileNotFoundError('Execute scripts/fetch_packaging_tools.py antes do empacotamento.')
    if hashlib.sha256((tools/'7zr.exe').read_bytes()).hexdigest() != 'e80db2dad715a6c5adabe292d31ed54e11f4e9b99ca349827b50573b175891bc':
        raise RuntimeError('O extrator 7-Zip não corresponde à versão oficial fixada.')
    version = json.loads((root/'desktop/package.json').read_text(encoding='utf-8'))['version']
    work = root / 'build/complete-bundle' / args.output.stem
    work.mkdir(parents=True, exist_ok=True)
    generated = work/'OfflineLauncher.cs'
    generated.write_text((root/'packaging/OfflineLauncher.cs').read_text(encoding='utf-8').replace('@APP_VERSION@', version), encoding='utf-8-sig')
    stub = work/'OfflineLauncher.exe'
    compiler = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler), '/nologo', '/target:winexe', '/platform:x64', '/optimize+',
                    '/reference:System.Windows.Forms.dll', '/reference:System.Drawing.dll',
                    f'/resource:{tools / "7zr.exe"},SevenZip', f'/resource:{tools / "License.txt"},SevenZipLicense',
                    f'/resource:{tools / "copying.txt"},SevenZipCopying',
                    f'/out:{stub}', str(generated), str(root/'packaging/ArchivePreparation.cs')], check=True)
    reserved = {'bundle-aliases.tsv', 'bundle-size.txt', '.payload.7z', '.extractor.exe'}
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.relative_to(source).as_posix() not in reserved)
    unique, aliases, names = {}, [], []
    total = 0
    for index, path in enumerate(files, 1):
        name = path.relative_to(source).as_posix()
        total += path.stat().st_size
        with path.open('rb') as stream:
            identity = hashlib.file_digest(stream, 'sha256').hexdigest()
        if identity in unique:
            aliases.append((name, unique[identity]))
        else:
            unique[identity] = name
            names.append(name)
        if index % 1000 == 0:
            print(f'Conferidos {index}/{len(files)} arquivos.', flush=True)
    (source/'bundle-aliases.tsv').write_text(''.join(f'{target}\t{original}\n' for target, original in aliases), encoding='utf-8')
    (source/'bundle-size.txt').write_text(str(total), encoding='utf-8')
    names += ['bundle-aliases.tsv', 'bundle-size.txt']
    listing = work/'files.txt'
    listing.write_text('\n'.join(names)+'\n', encoding='utf-8')
    archive = work/'complete.7z'
    if archive.exists():
        archive.unlink()
    print(f'Compactando {len(names)} entradas; {len(aliases)} duplicatas serão restauradas na preparação. Um núcleo e dicionário de 32 MiB.', flush=True)
    subprocess.run([str(tools/'7zr.exe'), 'a', '-t7z', '-mx=3', '-m0=lzma2:d=32m', '-mmt=1', '-ms=64m',
                    '-scsUTF-8', '-bsp0', str(archive), '@'+str(listing)], cwd=normal_source, check=True)
    if stub.stat().st_size + archive.stat().st_size + 64 >= 4 * 1024**3:
        raise RuntimeError('O pacote ainda excede o limite de 4 GiB de executáveis Windows.')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').digest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as target:
        with stub.open('rb') as stream:
            shutil.copyfileobj(stream, target, 1024*1024)
        offset = target.tell()
        with archive.open('rb') as stream:
            shutil.copyfileobj(stream, target, 1024*1024)
        target.write(struct.pack('<qqq32s8s', offset, archive.stat().st_size, total, digest, b'GER7Z001'))
    report = {'version': version, 'format': 'offline_executable_7z', 'output': str(args.output),
              'download_bytes': args.output.stat().st_size, 'installed_bytes': total,
              'file_count': len(files), 'deduplicated_files': len(aliases), 'payload_sha256': digest.hex(),
              'included_modes': ['cpu','nvidia','amd'] if not args.skip_engine_check else ['packager_test']}
    args.output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
