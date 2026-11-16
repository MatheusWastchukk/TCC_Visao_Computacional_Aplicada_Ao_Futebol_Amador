"""Atualiza o catálogo fixado usando índices oficiais e hashes da instalação AMD validada."""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.urls.extend(value for key, value in attrs if key == 'href')


def main():
    modes = {'nvidia': [], 'amd': []}
    for package, version in [('torch', '2.9.1'), ('torchvision', '0.24.1')]:
        index = f'https://download.pytorch.org/whl/cu126/{package}/'
        with urlopen(index, timeout=60) as response:
            links = Links(); links.feed(response.read().decode())
        filename = f'{package}-{version}+cu126-cp312-cp312-win_amd64.whl'
        link = next(urljoin(index, u) for u in links.urls if unquote(urlsplit(u).path).endswith('/'+filename))
        digest = urlsplit(link).fragment.removeprefix('sha256=')
        modes['nvidia'].append({'filename': filename, 'url': link.split('#')[0], 'sha256': digest})
    site = ROOT/'.venv-rocm/Lib/site-packages'
    for name in ['rocm_sdk_core-7.2.1', 'rocm_sdk_devel-7.2.1', 'rocm_sdk_libraries_custom-7.2.1',
                 'rocm-7.2.1', 'torch-2.9.1+rocm7.2.1', 'torchvision-0.24.1+rocm7.2.1']:
        provenance = json.loads((site/(name+'.dist-info')/'direct_url.json').read_text())
        url = provenance['url']
        assert url.startswith('https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/')
        modes['amd'].append({'filename': unquote(urlsplit(url).path.rsplit('/', 1)[1]),
                             'url': url, 'sha256': provenance['archive_info']['hashes']['sha256']})
    for packages in modes.values():
        for package in packages:
            assert len(package['sha256']) == 64
            with urlopen(Request(package['url'], method='HEAD'), timeout=60) as response:
                package['bytes'] = int(response.headers['Content-Length'])
            print(package['filename'], package['bytes'], flush=True)
    target = ROOT/'packaging/gpu_catalog.json'
    target.write_text(json.dumps({'schema': 1, 'python': 'cp312-win_amd64', 'modes': modes}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
