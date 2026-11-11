"""Prepara o compilador oficial Inno Setup em modo portátil na pasta de build."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parents[1]
folder=root/'build/tools'
folder.mkdir(parents=True,exist_ok=True)
target=folder/'innosetup-7.1.1-x64.exe'
url='https://github.com/jrsoftware/issrc/releases/download/is-7_1_1/innosetup-7.1.1-x64.exe'
expected='8a5a6711c3766a6e5fb74f280ba0e3551e719ce94538d901fdbd1f8f9920b3f0'
urlretrieve(url,target)
digest=hashlib.sha256(target.read_bytes()).hexdigest()
if digest != expected:
    raise RuntimeError('O compilador baixado não corresponde à versão oficial fixada.')
subprocess.run([str(target),'/PORTABLE=1','/VERYSILENT','/SUPPRESSMSGBOXES','/CURRENTUSER',
                '/NORESTART','/NOICONS','/DIR='+str(folder/'inno7'),
                '/LOG='+str(root/'build/inno_compiler_setup.log')],check=True)
(folder/'inno7/provenance.json').write_text(json.dumps({'url':url,'sha256':digest},indent=2)+'\n',encoding='utf-8')
print('Compilador portátil preparado em build/tools/inno7.',flush=True)
