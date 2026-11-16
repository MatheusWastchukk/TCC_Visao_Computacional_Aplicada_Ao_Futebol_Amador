"""Carrega o runtime GPU instalado antes dos módulos CPU congelados pelo PyInstaller."""
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys

from app.version import VERSION


def physical_runtime_folder(folder: Path) -> Path:
    """Usa o nome curto do próprio diretório, sem criar ou atravessar junções."""
    if os.name != 'nt':
        return folder
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    short_path = kernel.GetShortPathNameW
    short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    short_path.restype = wintypes.DWORD
    target = str(folder.absolute())
    required = short_path(target, None, 0)
    if not required:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required)
    written = short_path(target, buffer, required)
    if not written or written >= required:
        raise ctypes.WinError(ctypes.get_last_error())
    # Volumes sem nomes 8.3 podem devolver o caminho completo. A validação
    # real da GPU continua obrigatória para verificar esse ambiente.
    return Path(buffer.value)


class VendorFinder:
    def __init__(self, folder):
        self.folder = str(folder)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in {'torch', 'torchvision', 'rocm_sdk', '_rocm_sdk_core',
                                         '_rocm_sdk_devel', '_rocm_sdk_libraries_custom'}:
            return None
        spec = PathFinder.find_spec(fullname, path if path is not None else [self.folder])
        if spec is None:
            raise ModuleNotFoundError(f'O componente GPU está incompleto: {fullname}')
        return spec


def activate_vendor_runtime():
    configured = os.environ.get('GOAL_REVIEW_VENDOR_ROOT')
    if not configured:
        return
    folder = physical_runtime_folder(Path(configured))
    descriptor = json.loads((folder.resolve().parent/'runtime.json').read_text(encoding='utf-8'))
    mode = os.environ.get('GOAL_REVIEW_ACCELERATOR')
    if descriptor['schema'] != 1 or descriptor['app_version'] != VERSION or descriptor['mode'] != mode:
        raise RuntimeError('O componente GPU precisa corresponder à versão e ao modo do aplicativo.')
    if descriptor['folder'] != folder.resolve().name or not (folder/'torch/__init__.py').is_file():
        raise RuntimeError('O componente GPU está incompleto. Execute o instalador novamente.')
    sys.path.insert(0, str(folder))
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, VendorFinder(folder))
