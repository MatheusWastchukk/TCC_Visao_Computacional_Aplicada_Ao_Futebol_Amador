"""Verifica associação da GPU e agregação dos motores sem executar modelos."""
import os
import time

import pytest

from app.windows_gpu_metrics import WindowsGpuMetrics, busiest_engine, select_adapter


def row(pid, low, engine, value):
    return (f'pid_{pid}_luid_0x00000000_0x{low:08x}_phys_0_eng_{engine}_engtype_3D', value)


def test_uses_busiest_engine_of_selected_gpu_not_sum_of_all_engines():
    rows = [row(1, 10, 0, 25), row(2, 10, 0, 20), row(1, 10, 1, 40), row(3, 11, 0, 99)]
    assert busiest_engine(rows, (0, 10)) == 45


def test_zero_is_valid_but_missing_gpu_and_invalid_samples_are_unavailable():
    assert busiest_engine([row(1, 10, 0, 0)], (0, 10)) == 0
    assert busiest_engine([row(1, 11, 0, 99)], (0, 10)) is None
    assert busiest_engine([row(1, 10, 0, float('nan')), row(1, 10, 0, -1)], (0, 10)) is None
    assert busiest_engine([row(1, 10, 0, 110)], (0, 10)) == 100


def test_matches_selected_adapter_and_refuses_ambiguous_identical_cards():
    cards = [{'name': 'AMD Radeon RX 9060 XT', 'vendor_id': 0x1002, 'luid': (0, 10)},
             {'name': 'AMD Radeon Graphics', 'vendor_id': 0x1002, 'luid': (0, 11)}]
    assert select_adapter(cards, 'AMD Radeon RX 9060 XT', 'amd') == (0, 10)
    with pytest.raises(ValueError):
        select_adapter(cards, 'Nome desconhecido', 'amd')
    with pytest.raises(ValueError):
        select_adapter([cards[0], {**cards[0], 'luid': (0, 12)}], cards[0]['name'], 'amd')


class FakeQuery:
    def __init__(self, value):
        self.value, self.closed = value, False

    def read(self):
        return [row(1, 10, 0, self.value)]

    def close(self):
        self.closed = True


def wait_for(monitor, condition):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        value = monitor.sample('Radeon', 'amd')
        if condition(value): return value
        time.sleep(.005)
    raise AssertionError(value)


@pytest.mark.skipif(os.name != 'nt', reason='Monitor exclusivo do Windows.')
def test_background_sampler_keeps_query_and_publishes_real_zero():
    query = FakeQuery(0)
    calls = []
    def factory():
        calls.append(True)
        return query
    monitor = WindowsGpuMetrics(interval=.01, query_factory=factory,
        adapter_provider=lambda: [{'name': 'Radeon', 'vendor_id': 0x1002, 'luid': (0, 10)}])
    try:
        assert monitor.sample('Radeon', 'amd')['gpu_utilization_percent'] is None
        result = wait_for(monitor, lambda value: value['gpu_utilization_percent'] == 0)
        assert result['gpu_utilization_status'] == 'available'
        assert result['gpu_utilization_source'] == 'windows_wddm'
        for _ in range(20): assert monitor.sample('Radeon', 'amd')['gpu_utilization_percent'] == 0
        assert len(calls) == 1
    finally:
        monitor.close(); monitor.thread.join(timeout=1)
    assert query.closed


@pytest.mark.skipif(os.name != 'nt', reason='Monitor exclusivo do Windows.')
def test_transient_failure_does_not_disable_metrics_permanently():
    calls = []
    query = FakeQuery(36)
    def factory():
        calls.append(True)
        if len(calls) == 1: raise OSError('Contador ainda não disponível')
        return query
    monitor = WindowsGpuMetrics(interval=.01, retry_interval=.01, query_factory=factory,
        adapter_provider=lambda: [{'name': 'Radeon', 'vendor_id': 0x1002, 'luid': (0, 10)}])
    try:
        monitor.sample('Radeon', 'amd')
        result = wait_for(monitor, lambda value: value['gpu_utilization_percent'] == 36)
        assert result['gpu_utilization_error'] is None
        assert len(calls) >= 2
    finally:
        monitor.close(); monitor.thread.join(timeout=1)
