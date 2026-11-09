"""Regressão da CPU zerada e das consultas concorrentes que compartilham a amostra."""
from types import SimpleNamespace

from app.process_metrics import ProcessMetrics


class FakeProcess:
    def __init__(self):
        self.calls = 0

    def cpu_percent(self, interval=None):
        self.calls += 1
        return 0.0 if self.calls == 1 else 200.0

    def memory_info(self):
        return SimpleNamespace(rss=100 * 1024**2)

    def memory_percent(self):
        return 1.0


def test_cpu_preserves_previous_measurement_and_normalizes_capacity(monkeypatch):
    monkeypatch.setattr("app.process_metrics.psutil.cpu_count", lambda: 4)
    clock = [0.0]
    process = FakeProcess()
    monitor = ProcessMetrics(process=process, clock=lambda: clock[0])
    assert monitor.sample()["process_cpu_percent"] is None
    clock[0] = 1.0
    assert monitor.sample()["process_cpu_percent"] == 50.0
    for _ in range(20):
        assert monitor.sample()["process_cpu_percent"] == 50.0
    assert process.calls == 2
    clock[0] = 2.0
    assert monitor.sample()["process_ram_mb"] == 100.0
    assert process.calls == 3
