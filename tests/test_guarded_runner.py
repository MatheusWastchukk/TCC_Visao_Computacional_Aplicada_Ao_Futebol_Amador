"""Verifica a proteção de RAM e a limpeza de subprocessos no Windows."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Supervisor do pacote Windows.')
ROOT = Path(__file__).resolve().parents[1]


def run_guarded(folder, *command, minimum='3'):
    report = folder/'report.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_guarded.py'), '--min-free-gb', minimum,
                             '--report', str(report), '--', *command], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    return result, json.loads(report.read_text(encoding='utf-8'))


def test_memory_precheck_does_not_start_child(tmp_path):
    marker = tmp_path/'should-not-exist'
    command = f'from pathlib import Path; Path({str(marker)!r}).touch()'
    result, report = run_guarded(tmp_path, sys.executable, '-c', command, minimum='9999')
    assert result.returncode == 3
    assert report['status'] == 'stopped_low_memory'
    assert not marker.exists()


def test_job_closes_descendant_after_parent_exits(tmp_path):
    marker = tmp_path/'child.pid'
    command = ('import subprocess,sys,time; from pathlib import Path; time.sleep(.5); '
               'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],'
               'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
               f'Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(1)')
    result, report = run_guarded(tmp_path, sys.executable, '-c', command)
    assert result.returncode == 0, result.stderr
    assert report['status'] == 'completed'
    pid = int(marker.read_text())
    deadline = time.monotonic()+5
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(.1)
    alive = psutil.pid_exists(pid)
    if alive:
        psutil.Process(pid).kill()
    assert not alive, 'O subprocesso sobreviveu à conclusão da tarefa supervisionada.'
