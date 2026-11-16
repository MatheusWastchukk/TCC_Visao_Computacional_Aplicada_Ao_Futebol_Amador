"""Testa o instalador online em build, preservando instalações e dados externos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import winreg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('install', 'verify', 'uninstall'))
    parser.add_argument('--mode', choices=('cpu', 'amd', 'nvidia'), default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root/'build/online_installer_e2e'
    app = work/'app'
    assert app.resolve().is_relative_to((root/'build').resolve())
    work.mkdir(parents=True, exist_ok=True)
    version = json.loads((root/'desktop/package.json').read_text())['version']
    key = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\br.edu.tcc.goaleventreview.online_is1'

    def registration():
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
                location = Path(winreg.QueryValueEx(handle, 'InstallLocation')[0])
                uninstaller = Path(winreg.QueryValueEx(handle, 'UninstallString')[0].strip('"'))
            if location.resolve() != app.resolve() or not uninstaller.resolve().is_relative_to(app.resolve()):
                raise RuntimeError('Preservar a instalação existente fora da pasta isolada de teste.')
            return uninstaller
        except FileNotFoundError:
            return None

    previous = registration()
    if args.step == 'install':
        installer = root/f'dist/online-installer/GoalEventReview-{version}-Online-Setup-x64.exe'
        subprocess.run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                        '/GPU='+args.mode, '/DIR='+str(app), '/TASKS=',
                        '/LOG='+str(work/f'install_{args.mode}.log')], check=True, timeout=3600)
        assert registration() is not None
        assert (app/'Goal Event Review.exe').is_file()
        (app/'user-file.txt').write_text('Preservar arquivo adicional.', encoding='utf-8')
        (work/'session-marker.txt').write_text('Preservar dados externos.', encoding='utf-8')
        if args.mode != 'cpu':
            descriptor = json.loads((app/f'resources/gpu/{args.mode}/runtime.json').read_text())
            assert descriptor['mode'] == args.mode and descriptor['app_version'] == version
            payload = app/f'resources/gpu/{args.mode}'/descriptor['folder']
            assert (payload/'torch/__init__.py').is_file()
            (payload/'user-note.txt').write_text('Preservar arquivo não distribuído.', encoding='utf-8')
        result = {'ok': True, 'mode': args.mode, 'upgrade_in_test_folder': previous is not None}
        output = f'install_{args.mode}_result.json'
    elif args.step == 'verify':
        checked = 0
        source = root/'dist/cpu/win-unpacked'
        for original in source.rglob('*'):
            if not original.is_file(): continue
            target = app/original.relative_to(source)
            with original.open('rb') as before, target.open('rb') as after:
                assert hashlib.file_digest(before, 'sha256').digest() == hashlib.file_digest(after, 'sha256').digest(), str(target)
            checked += 1
        assert (app/'resources/gpu_catalog.json').read_bytes() == (root/'packaging/gpu_catalog.json').read_bytes()
        result = {'ok': True, 'base_files_verified_by_sha256': checked}
        output = 'integrity_result.json'
    else:
        assert previous is not None
        gpu_files = []
        preserved = []
        gpu = Path('\\\\?\\'+str(app.resolve()))/'resources/gpu'
        for record_path in gpu.glob('*/payload-*/installed-files.json'):
            record = json.loads(record_path.read_text())
            gpu_files.extend(record_path.parent/name for name in record['files'])
            preserved.append(record_path.parent/'user-note.txt')
        subprocess.run([str(previous), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                        '/LOG='+str(work/'uninstall.log')], check=True, timeout=300)
        assert registration() is None
        for original in (root/'dist/cpu/win-unpacked').rglob('*'):
            if original.is_file(): assert not (app/original.relative_to(root/'dist/cpu/win-unpacked')).exists()
        assert all(not path.exists() for path in gpu_files)
        assert all(path.is_file() for path in preserved)
        assert (app/'user-file.txt').is_file() and (work/'session-marker.txt').is_file()
        result = {'ok': True, 'gpu_files_removed': len(gpu_files), 'user_files_preserved': True}
        output = 'uninstall_result.json'
    (work/output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
