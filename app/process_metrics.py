"""Amostragem não bloqueante do processo, compartilhada entre as threads da API."""
from __future__ import annotations

import threading
import time

import psutil


class ProcessMetrics:
    """Mantém a referência anterior e limita a frequência de atualização da CPU."""

    def __init__(self, process=None, clock=time.monotonic, interval: float = 1.0):
        self.process = process if process is not None else psutil.Process()
        self.clock = clock
        self.interval = interval
        self.lock = threading.Lock()
        self.cpu_count = max(1, psutil.cpu_count() or 1)
        self.cpu_percent: float | None = None
        # A primeira leitura do psutil apenas estabelece a referência temporal.
        self.process.cpu_percent(interval=None)
        self.last_sample = self.clock()

    def sample(self) -> dict[str, object]:
        with self.lock:
            now = self.clock()
            if now - self.last_sample >= self.interval:
                value = self.process.cpu_percent(interval=None) / self.cpu_count
                self.cpu_percent = round(max(0.0, min(100.0, value)), 1)
                self.last_sample = now
            return {
                "process_cpu_percent": self.cpu_percent,
                "process_cpu_scale": "total_machine_capacity",
                "logical_cpu_count": self.cpu_count,
                "process_ram_mb": round(self.process.memory_info().rss / 1024**2, 1),
                "process_ram_percent": round(self.process.memory_percent(), 2),
            }
