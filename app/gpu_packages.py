"""Instala pacotes oficiais fixados, sem pip, scripts de instalação ou acesso à rede."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
from urllib.parse import urlsplit
import uuid
import zipfile

from app.version import VERSION


def native_path(path: Path) -> Path:
    value = str(path.resolve())
    return Path('\\\\?\\' + value) if os.name == 'nt' and not value.startswith('\\\\?\\') else Path(value)


def relative_file(name: str) -> Path:
    value = PurePosixPath(name.replace('\\', '/'))
    if not value.parts or value.is_absolute() or '..' in value.parts or ':' in name:
        raise ValueError(f'Caminho inválido no pacote: {name}')
    return Path(*value.parts)


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if catalog['schema'] != 1 or catalog['python'] != 'cp312-win_amd64':
        raise ValueError('Catálogo de GPU incompatível com este aplicativo.')
    for mode in ('nvidia', 'amd'):
        for package in catalog['modes'][mode]:
            host = urlsplit(package['url'])
            allowed = {'download.pytorch.org', 'download-r2.pytorch.org'} if mode == 'nvidia' else {'repo.radeon.com'}
            if host.scheme != 'https' or host.hostname not in allowed or host.username or host.password:
                raise ValueError('O catálogo deve usar exclusivamente fontes oficiais por HTTPS.')
            if relative_file(package['filename']).name != package['filename']:
                raise ValueError('Nome de download inválido.')
            if len(package['sha256']) != 64 or any(c not in '0123456789abcdef' for c in package['sha256']):
                raise ValueError('Hash SHA-256 inválido no catálogo.')
    return catalog


def extract_package(archive: Path, destination: Path) -> list[str]:
    """Extrai arquivos regulares com buffer limitado e rejeita caminhos externos e links."""
    files = []

    def write(name, source):
        relative = relative_file(name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open('wb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        files.append(relative.as_posix())

    if archive.name.endswith('.whl'):
        with zipfile.ZipFile(archive) as wheel:
            for member in wheel.infolist():
                relative_file(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Links não são permitidos nos pacotes.')
                if not member.is_dir():
                    write(member.filename, wheel.open(member))
    else:
        # O pequeno pacote oficial rocm contém Python puro em src/rocm_sdk.
        # Não executamos setup.py nem instalamos compiladores ou dependências extras.
        with tarfile.open(archive, 'r:gz') as source:
            for member in source:
                relative_file(member.name)
                if member.issym() or member.islnk():
                    raise ValueError('Links não são permitidos nos pacotes.')
                parts = PurePosixPath(member.name).parts
                if member.isfile() and len(parts) > 3 and parts[1:3] == ('src', 'rocm_sdk'):
                    write('/'.join(parts[2:]), source.extractfile(member))
        if 'rocm_sdk/__init__.py' not in files:
            raise ValueError('O pacote oficial ROCm não contém o módulo esperado.')
    return files


def install_runtime(mode: str, downloads: Path, gpu_root: Path, catalog_path: Path) -> dict:
    catalog = load_catalog(catalog_path)
    packages = catalog['modes'][mode]
    # Verifica todos os downloads antes de modificar o destino.
    for package in packages:
        path = downloads / package['filename']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != package['bytes'] or digest != package['sha256']:
            raise ValueError(f'Download incompleto ou corrompido: {package["filename"]}')
    identity = hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()[:16]
    parent = native_path(gpu_root) / mode
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / ('payload-' + identity)
    stage = parent / ('staging-' + uuid.uuid4().hex)
    if not target.exists():
        stage.mkdir()
        try:
            files = []
            for package in packages:
                print('Preparando '+package['filename'], flush=True)
                files.extend(extract_package(downloads/package['filename'], stage))
            record = {'payload_id': identity, 'files': sorted(set(files))}
            (stage/'installed-files.json').write_text(json.dumps(record), encoding='utf-8')
            stage.rename(target)
        finally:
            if stage.exists():
                assert stage.parent == parent and stage.name.startswith('staging-')
                shutil.rmtree(stage)
    record = json.loads((target/'installed-files.json').read_text(encoding='utf-8'))
    if record['payload_id'] != identity:
        raise ValueError('A pasta do componente está ocupada por outra instalação.')
    descriptor = {'schema': 1, 'mode': mode, 'app_version': VERSION, 'folder': target.name,
                  'payload_id': identity, 'packages': packages}
    temporary = parent/'runtime.json.new'
    temporary.write_text(json.dumps(descriptor, indent=2), encoding='utf-8')
    os.replace(temporary, parent/'runtime.json')
    return descriptor


def remove_runtimes(gpu_root: Path) -> None:
    """Remove somente arquivos registrados; preserva arquivos adicionais e pastas ocupadas."""
    root = native_path(gpu_root)
    for mode in ('nvidia', 'amd'):
        parent = root/mode
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('A pasta do componente não pode ser um link.')
        for payload in parent.glob('payload-*'):
            if payload.is_symlink() or payload.is_junction():
                raise ValueError('O componente não pode apontar para outra pasta.')
            record_path = payload/'installed-files.json'
            record = json.loads(record_path.read_text(encoding='utf-8'))
            directories = set()
            for name in record['files']:
                relative = relative_file(name)
                path = payload/relative
                if not path.resolve().is_relative_to(payload.resolve()):
                    raise ValueError('Arquivo fora da pasta do componente.')
                path.unlink(missing_ok=True)
                directories.update(p for p in path.parents if p.is_relative_to(payload))
            record_path.unlink()
            for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                try: directory.rmdir()
                except OSError: pass
        (parent/'runtime.json').unlink(missing_ok=True)
        for directory in (parent, root):
            try: directory.rmdir()
            except OSError: pass
