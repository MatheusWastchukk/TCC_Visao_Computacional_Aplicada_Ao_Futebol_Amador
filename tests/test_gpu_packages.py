"""Verifica integridade, isolamento e remoção dos componentes opcionais."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from app.gpu_packages import extract_package, install_runtime, remove_runtimes
from app.vendor_runtime import VendorFinder
from app.version import VERSION


def fixture_package(tmp_path):
    archive = tmp_path/'torch.whl'
    with zipfile.ZipFile(archive, 'w') as wheel:
        wheel.writestr('torch/__init__.py', 'value = 42\n')
        wheel.writestr('torch/version.py', 'cuda = "12.6"\n')
    package = {'filename': archive.name, 'url': 'https://download.pytorch.org/whl/torch.whl',
               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size}
    catalog = tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'schema': 1, 'python': 'cp312-win_amd64',
                                  'modes': {'nvidia': [package], 'amd': []}}))
    return archive, catalog


def test_corrupted_download_does_not_publish_runtime(tmp_path):
    archive, catalog = fixture_package(tmp_path)
    archive.write_bytes(b'download interrompido')
    with pytest.raises(ValueError, match='corrompido'):
        install_runtime('nvidia', tmp_path, tmp_path/'gpu', catalog)
    assert not (tmp_path/'gpu').exists()


def test_install_is_atomic_and_uninstall_preserves_user_files(tmp_path):
    archive, catalog = fixture_package(tmp_path)
    root = tmp_path/'gpu'
    descriptor = install_runtime('nvidia', tmp_path, root, catalog)
    payload = root/'nvidia'/descriptor['folder']
    assert (payload/'torch/version.py').is_file()
    assert json.loads((root/'nvidia/runtime.json').read_text()) == descriptor
    (payload/'user.txt').write_text('preservar')
    outside = tmp_path/'outside.txt'; outside.write_text('preservar')
    remove_runtimes(root)
    assert not (payload/'torch/version.py').exists()
    assert (payload/'user.txt').is_file() and outside.is_file()


@pytest.mark.parametrize('name', ['../escape', 'C:/escape', '/escape', 'a/../../escape'])
def test_package_cannot_write_outside_destination(tmp_path, name):
    archive = tmp_path/'bad.whl'
    with zipfile.ZipFile(archive, 'w') as wheel:
        wheel.writestr(name, b'bad')
    with pytest.raises(ValueError, match='Caminho'):
        extract_package(archive, tmp_path/'destination')


def test_gpu_finder_uses_downloaded_modules_before_frozen_cpu(tmp_path):
    archive, catalog = fixture_package(tmp_path)
    extract_package(archive, tmp_path/'vendor')
    finder = VendorFinder(tmp_path/'vendor')
    spec = finder.find_spec('torch')
    assert Path(spec.origin) == tmp_path/'vendor/torch/__init__.py'
    assert finder.find_spec('json') is None
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec('torch.absent', [str(tmp_path/'vendor/torch')])


@pytest.mark.skipif(os.name != 'nt', reason='Proteção RedirectionGuard do Windows.')
def test_physical_runtime_loads_with_redirection_guard_enforced(tmp_path):
    parent = tmp_path/'amd'
    payload = parent/'payload-0123456789abcdef'
    (payload/'torch').mkdir(parents=True)
    (payload/'torch/__init__.py').write_text('')
    (parent/'runtime.json').write_text(json.dumps({'schema': 1, 'mode': 'amd',
        'app_version': VERSION, 'folder': payload.name}))
    code = '''
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
from app.vendor_runtime import activate_vendor_runtime, VendorFinder
import sys
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.SetProcessMitigationPolicy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
policy = wintypes.DWORD(1)
if not kernel.SetProcessMitigationPolicy(16, ctypes.byref(policy), ctypes.sizeof(policy)):
    raise ctypes.WinError(ctypes.get_last_error())
activate_vendor_runtime()
finder = next(item for item in sys.meta_path if isinstance(item, VendorFinder))
assert Path(finder.folder).resolve() == Path(os.environ['GOAL_REVIEW_VENDOR_ROOT']).resolve()
assert finder.find_spec('torch') is not None
print('RedirectionGuard ativo: runtime físico acessível.')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'GOAL_REVIEW_VENDOR_ROOT': str(payload), 'GOAL_REVIEW_ACCELERATOR': 'amd'},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
