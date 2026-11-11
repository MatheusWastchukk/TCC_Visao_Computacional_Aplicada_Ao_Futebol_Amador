"""Gera um instalador offline reutilizando a compactação completa validada."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in value or '\n' in value or '\r' in value:
        raise ValueError(f'Caminho inválido no manifesto: {value}')
    return path


def quoted(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=Path('build/tools/inno7/ISCC.exe'))
    parser.add_argument('--source', type=Path, default=Path('dist/complete/win-unpacked'))
    parser.add_argument('--fixture-root', type=Path, help='Uso exclusivo na validação com dados artificiais pequenos.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = json.loads((root/'desktop/package.json').read_text(encoding='utf-8'))['version']
    fixture = args.fixture_root.resolve() if args.fixture_root else None
    archive = fixture/'complete.7z' if fixture else root/f'build/complete-bundle/GoalEventReview-{version}-offline-x64/complete.7z'
    report_path=fixture/'package.json' if fixture else root/f'dist/complete/GoalEventReview-{version}-offline-x64.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['version'] != version or set(report['included_modes']) != {'cpu','amd','nvidia'}:
        raise RuntimeError('O arquivo completo precisa corresponder à versão e incluir os três ambientes.')
    extractor=root/'build/tools/7zip/7zr.exe'
    if hashlib.sha256(extractor.read_bytes()).hexdigest() != 'e80db2dad715a6c5adabe292d31ed54e11f4e9b99ca349827b50573b175891bc':
        raise RuntimeError('O extrator incluído não corresponde à versão oficial fixada.')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != report['payload_sha256']:
        raise RuntimeError('A compactação completa não passou na verificação SHA-256.')
    source = Path('\\\\?\\'+str(fixture/'source' if fixture else args.source.resolve()))
    for mode, folder in (('cpu','backend'),('nvidia','engines/nvidia'),('amd','engines/amd')):
        component=json.loads((source/'resources'/folder/'component.json').read_text(encoding='utf-8-sig'))
        if component['mode'] != mode or component['app_version'] != version:
            raise RuntimeError(f'O componente {mode} não pertence à versão {version}.')
    work = fixture/'work' if fixture else root/'build/installer'
    work.mkdir(parents=True,exist_ok=True)
    aliases=[]; alias_bytes=0
    for line in (source/'bundle-aliases.tsv').read_text(encoding='utf-8').splitlines():
        target, original = (safe_relative(v) for v in line.split('\t'))
        size=(source/str(original)).stat().st_size
        alias_bytes += size
        aliases.append((target,original))
    unique_names={safe_relative(n) for n in (archive.parent/'files.txt').read_text(encoding='utf-8').splitlines()}
    unique_bytes=sum((source/str(name)).stat().st_size for name in unique_names)
    packaged_names=unique_names | {target for target,original in aliases}
    directories={parent for name in packaged_names for parent in name.parents if str(parent)!='.'}
    uninstall_entries=[f'Type: files; Name: {quoted("{app}"+chr(92)+str(PureWindowsPath(name)))}' for name in sorted(packaged_names)]
    uninstall_entries += [f'Type: dirifempty; Name: {quoted("{app}"+chr(92)+str(PureWindowsPath(name)))}' for name in sorted(directories,key=lambda p:(-len(p.parts),str(p)))]
    uninstall_include=work/'uninstall.iss'
    uninstall_include.write_text('\n'.join(uninstall_entries)+'\n',encoding='utf-8-sig')
    output=fixture/'output' if fixture else root/'dist/installer'
    output.mkdir(parents=True,exist_ok=True)
    guide=root/'docs/guia_teste_instalador.txt'
    parts=work/'parts'
    parts.mkdir(exist_ok=True)
    entries=[]
    chunk_limit=16384 if fixture else 128*1024**2
    with archive.open('rb') as stream:
        for index in range(1, 10000):
            block=stream.read(min(1024**2,chunk_limit))
            if not block:
                break
            part=parts/f'goal-review.7z.{index:03d}'
            with part.open('wb') as target:
                remaining=chunk_limit
                while block:
                    target.write(block); remaining-=len(block)
                    if remaining <= 0: break
                    block=stream.read(min(1024**2,remaining))
                target.flush(); os.fsync(target.fileno())
            entries.append(f'Source: {quoted(part)}; DestDir: "{{tmp}}"; DestName: {quoted(part.name)}; Flags: deleteafterinstall nocompression')
    archive_entries=work/'archive_parts.iss'
    entries[-1] += '; AfterInstall: PrepareComponents'
    archive_entries.write_text('\n'.join(entries)+'\n',encoding='utf-8-sig')
    definitions={'AppVersion':version,'OutputFolder':output,'ArchiveEntries':archive_entries,'UninstallEntries':uninstall_include,
                 'SevenZipBinary':extractor,
                 'InstallerAppId':'br.edu.tcc.goaleventreview.fixture' if fixture else 'br.edu.tcc.goaleventreview.complete',
                 'SevenZipLicense':root/'build/tools/7zip/License.txt',
                 'SevenZipCopying':root/'build/tools/7zip/copying.txt',
                 'SevenZipNotice':root/'packaging/third_party_7zip.txt','TestingGuide':guide}
    script=work/'setup.iss'
    script.write_text('\n'.join(f'#define {key} {quoted(value)}' for key,value in definitions.items())+
                      f'\n#define UniqueBytes {unique_bytes}\n#define InstalledBytes {report["installed_bytes"]}\n#define PartCount {len(entries)}\n#include {quoted(root/"packaging/windows_installer.iss")}\n',encoding='utf-8-sig')
    subprocess.run([str(args.compiler.resolve()),str(script)],check=True)
    executable=output/f'GoalEventReview-{version}-Setup-x64.exe'
    with executable.open('rb') as stream:
        executable_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    info={'version':version,'format':'windows_installer_inno7','installer':str(executable),
          'installer_bytes':executable.stat().st_size,'installed_bytes':report['installed_bytes'],
          'payload_sha256':digest,'sha256':executable_hash,'included_modes':report['included_modes'],
          'aliases':len(aliases),'alias_bytes':alias_bytes,'unique_bytes':unique_bytes}
    executable.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'SHA256SUMS.txt').write_text(f'{executable_hash}  {executable.name}\n',encoding='utf-8')
    (output/'LEIA-ME-TESTE.txt').write_text(guide.read_text(encoding='utf-8'),encoding='utf-8-sig')
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
