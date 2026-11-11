"""Executa uma tarefa local por vez e a interrompe se faltar memória."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-free-gb', type=float, default=2.5)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Informe o comando após --.')
    root = Path(__file__).resolve().parents[1]
    folder = root / 'build/guard'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'active.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        try:
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('Outra tarefa supervisionada já está em execução.', flush=True)
            lock.close()
            return 2
    started = time.monotonic()
    report = {'command': command, 'minimum_free_gb': None, 'status': 'starting',
              'minimum_required_free_gb': args.min_free_gb}
    child = None
    job = None

    def stop_child():
        if child is None or child.poll() is not None:
            return
        if job is not None:
            job.terminate()
        elif os.name == 'nt':
            subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            child.kill()
        child.wait(timeout=20)

    try:
        last_print = -30.0
        while True:
            free = psutil.virtual_memory().available / 1024**3
            report['minimum_free_gb'] = min(free, report['minimum_free_gb'] or free)
            if free < args.min_free_gb:
                report['status'] = 'stopped_low_memory'
                print(f'Tarefa interrompida: apenas {free:.2f} GiB de RAM disponíveis. Limite: {args.min_free_gb:.2f} GiB.', flush=True)
                stop_child()
                return 3
            if child is None:
                flags = subprocess.CREATE_NO_WINDOW | 0x00000004 if os.name == 'nt' else 0
                env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
                child = subprocess.Popen(command, cwd=root, env=env, creationflags=flags, stdout=sys.stdout, stderr=sys.stderr)
                if os.name == 'nt':
                    from windows_job import WindowsJob
                    job = WindowsJob(child)
                try:
                    psutil.Process(child.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                if job is not None:
                    job.resume(child)
                report['pid'] = child.pid
            code = child.poll()
            if code is not None:
                report.update(status='completed' if code == 0 else 'failed', exit_code=code)
                return code
            now = time.monotonic() - started
            if now - last_print >= 30:
                scans = sum(p.info['name'] == 'rg.exe' for p in psutil.process_iter(['name']))
                report['last_search_process_count'] = scans
                if scans > 100:
                    report['status'] = 'stopped_indexing_overload'
                    print(f'Tarefa interrompida: {scans} processos de busca acumulados no editor.', flush=True)
                    stop_child()
                    return 4
                print(f'Supervisão: {free:.2f} GiB de RAM disponíveis; {scans} buscas; uma tarefa em execução.', flush=True)
                last_print = now
            time.sleep(1)
    except BaseException:
        report['status'] = 'interrupted'
        stop_child()
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if job is not None:
            job.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
