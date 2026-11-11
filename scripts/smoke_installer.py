"""Valida instalação, integridade e remoção em uma pasta isolada de build."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import winreg


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('install','verify','uninstall'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    version=json.loads((root/'desktop/package.json').read_text(encoding='utf-8'))['version']
    work=root/'build/installer_e2e'
    app=work/'app'
    if not app.resolve().is_relative_to((root/'build').resolve()):
        raise RuntimeError('O diretório de teste precisa permanecer dentro de build.')
    key=r'Software\Microsoft\Windows\CurrentVersion\Uninstall\br.edu.tcc.goaleventreview.complete_is1'
    installer=root/f'dist/installer/GoalEventReview-{version}-Setup-x64.exe'

    def registration():
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key) as handle:
                return winreg.QueryValueEx(handle,'InstallLocation')[0]
        except FileNotFoundError:
            return None

    def extended(path):
        return Path('\\\\?\\'+str(path.resolve()))

    def registered_uninstaller():
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key) as handle:
            result=Path(winreg.QueryValueEx(handle,'UninstallString')[0].strip('"'))
        if not result.resolve().is_relative_to(app.resolve()):
            raise RuntimeError('O desinstalador registrado não pertence à pasta de teste.')
        return result

    if args.step=='install':
        if registration() is not None:
            raise RuntimeError('Já existe uma instalação com este identificador; preserve a instalação existente.')
        work.mkdir(parents=True,exist_ok=True)
        metadata=json.loads(installer.with_suffix('.json').read_text(encoding='utf-8'))
        group='Goal Event Review Teste '+metadata['sha256'][:8]
        subprocess.run([str(installer),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/CURRENTUSER',
                        '/DIR='+str(app),'/GROUP='+group,'/TASKS=','/LOG='+str(work/'install.log')],check=True,timeout=1200)
        assert Path(registration()).resolve()==app.resolve()
        assert (app/'Goal Event Review.exe').is_file() and registered_uninstaller().is_file()
        (app/'user-file.txt').write_text('Preservar arquivo não instalado.',encoding='utf-8')
        (work/'session-marker.txt').write_text('Preservar dados fora da instalação.',encoding='utf-8')
        result={'ok':True,'app':str(app),'start_menu_group':group}
    else:
        source=extended(root/'dist/complete/win-unpacked'); target=extended(app)
        if args.step=='verify':
            checked=0
            for original in source.rglob('*'):
                if not original.is_file(): continue
                relative=original.relative_to(source)
                installed=target/relative
                assert installed.is_file(),str(relative)
                with original.open('rb') as before, installed.open('rb') as after:
                    assert hashlib.file_digest(before,'sha256').digest()==hashlib.file_digest(after,'sha256').digest(),str(relative)
                checked+=1
            result={'ok':True,'files_verified_by_sha256':checked}
        else:
            assert registration() and Path(registration()).resolve()==app.resolve()
            subprocess.run([str(registered_uninstaller()),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',
                            '/LOG='+str(work/'uninstall.log')],check=True,timeout=1200)
            for original in source.rglob('*'):
                if original.is_file(): assert not (target/original.relative_to(source)).exists(),str(original)
            assert (app/'user-file.txt').is_file() and (work/'session-marker.txt').is_file()
            assert registration() is None
            result={'ok':True,'unregistered':True,'user_files_preserved':True}
    output={'install':'install','verify':'integrity','uninstall':'uninstall'}[args.step]
    (work/f'{output}_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
