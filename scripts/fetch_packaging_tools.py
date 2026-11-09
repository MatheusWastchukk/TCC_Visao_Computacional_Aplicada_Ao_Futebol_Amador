"""Obtém o extrator standalone oficial incluído no pacote offline."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json

root = Path(__file__).resolve().parents[1] / 'build/tools/7zip'
root.mkdir(parents=True, exist_ok=True)
urls = {
    '7zr.exe': 'https://github.com/ip7z/7zip/releases/download/26.06/7zr.exe',
    'License.txt': 'https://raw.githubusercontent.com/ip7z/7zip/26.06/DOC/License.txt',
    'copying.txt': 'https://raw.githubusercontent.com/ip7z/7zip/26.06/DOC/copying.txt',
}
report = {}
for name, url in urls.items():
    target = root / name
    urlretrieve(url, target)
    report[name] = {'url': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
(root/'sources.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
print(json.dumps(report), flush=True)
