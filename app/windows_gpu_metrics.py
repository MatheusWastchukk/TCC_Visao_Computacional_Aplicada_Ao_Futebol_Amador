"""Uso da GPU selecionada pelos contadores WDDM, sem depender de NVML/ROCm SMI."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import re
import threading
import time
import uuid


class _Guid(ctypes.Structure):
    _fields_ = [('data1', wintypes.DWORD), ('data2', wintypes.WORD),
                ('data3', wintypes.WORD), ('data4', ctypes.c_ubyte * 8)]


class _Luid(ctypes.Structure):
    _fields_ = [('low', wintypes.DWORD), ('high', wintypes.LONG)]


class _AdapterDesc(ctypes.Structure):
    _fields_ = [('description', wintypes.WCHAR * 128), ('vendor', wintypes.UINT),
                ('device', wintypes.UINT), ('subsystem', wintypes.UINT), ('revision', wintypes.UINT),
                ('video_memory', ctypes.c_size_t), ('system_memory', ctypes.c_size_t),
                ('shared_memory', ctypes.c_size_t), ('luid', _Luid)]


def _method(pointer, index, result, *arguments):
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[index])


def adapters() -> list[dict]:
    """Identifica nome, fornecedor e LUID físicos, sem presumir a ordem das GPUs."""
    library = ctypes.WinDLL('dxgi')
    create = library.CreateDXGIFactory
    create.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    identifier = _Guid.from_buffer_copy(uuid.UUID('7b7166ec-21c7-44ae-b21a-c9ae321ae369').bytes_le)
    factory = ctypes.c_void_p()
    if create(ctypes.byref(identifier), ctypes.byref(factory)) < 0:
        raise OSError('Não foi possível identificar os adaptadores de vídeo pelo Windows.')
    found = []
    try:
        enumerate_adapter = _method(factory, 7, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            result = enumerate_adapter(factory, index, ctypes.byref(adapter))
            if result & 0xffffffff == 0x887A0002:
                break
            if result < 0:
                raise OSError(f'Falha ao enumerar GPU: 0x{result & 0xffffffff:08x}.')
            try:
                description = _AdapterDesc()
                get_description = _method(adapter, 8, ctypes.c_long, ctypes.POINTER(_AdapterDesc))
                if get_description(adapter, ctypes.byref(description)) < 0:
                    continue
                found.append({'name': description.description, 'vendor_id': description.vendor,
                              'luid': (description.luid.high & 0xffffffff, description.luid.low)})
            finally:
                _method(adapter, 2, wintypes.ULONG)(adapter)
    finally:
        _method(factory, 2, wintypes.ULONG)(factory)
    return found


def select_adapter(found: list[dict], device_name: str, vendor: str) -> tuple[int, int]:
    normalize = lambda name: ''.join(c.lower() for c in name if c.isalnum())
    matching = [item for item in found if normalize(item['name']) == normalize(device_name)]
    if not matching:
        vendor_id = {'amd': 0x1002, 'nvidia': 0x10de}.get(vendor)
        matching = [item for item in found if vendor_id is not None and item['vendor_id'] == vendor_id]
    if len(matching) != 1:
        raise ValueError('Não foi possível associar com segurança a GPU selecionada aos contadores do Windows.')
    return matching[0]['luid']


_ENGINE = re.compile(r'luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_', re.I)


def busiest_engine(rows: list[tuple[str, float]], luid: tuple[int, int]) -> float | None:
    """Soma processos por motor da GPU e usa o motor mais ocupado, como o Windows."""
    engines = {}
    for name, value in rows:
        match = _ENGINE.search(name)
        if not match or (int(match[1], 16), int(match[2], 16)) != luid:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        key = (int(match[3]), int(match[4]))
        engines[key] = engines.get(key, 0.0) + value
    return round(min(100.0, max(engines.values())), 1) if engines else None


class _Value(ctypes.Structure):
    _fields_ = [('status', wintypes.DWORD), ('value', ctypes.c_double)]


class _Item(ctypes.Structure):
    _fields_ = [('name', wintypes.LPWSTR), ('formatted', _Value)]


class PdhGpuQuery:
    """Mantém a consulta entre amostras para calcular percentuais reais de atividade."""

    def __init__(self):
        self.library = ctypes.WinDLL('pdh')
        self.query = wintypes.HANDLE()
        self.counter = wintypes.HANDLE()
        definitions = {
            'PdhOpenQueryW': [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            'PdhAddEnglishCounterW': [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            'PdhCollectQueryData': [wintypes.HANDLE],
            'PdhCloseQuery': [wintypes.HANDLE],
            'PdhGetFormattedCounterArrayW': [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p],
        }
        for name, arguments in definitions.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, wintypes.DWORD
        try:
            self._check(self.library.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
            self._check(self.library.PdhAddEnglishCounterW(self.query,
                r'\GPU Engine(*)\Utilization Percentage', 0, ctypes.byref(self.counter)))
            self._check(self.library.PdhCollectQueryData(self.query))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status:
            raise OSError(f'Contador de GPU do Windows indisponível (0x{status:08x}).')

    def read(self) -> list[tuple[str, float]]:
        self._check(self.library.PdhCollectQueryData(self.query))
        size, count = wintypes.DWORD(), wintypes.DWORD()
        function = self.library.PdhGetFormattedCounterArrayW
        status = function(self.counter, 0x200 | 0x8000, ctypes.byref(size), ctypes.byref(count), None)
        if status != 0x800007D2:  # PDH_MORE_DATA
            self._check(status)
        if not size.value:
            return []
        if size.value > 16 * 1024**2:
            raise OSError('A consulta de GPU excedeu o limite de memória do monitor.')
        buffer = ctypes.create_string_buffer(size.value)
        self._check(function(self.counter, 0x200 | 0x8000, ctypes.byref(size), ctypes.byref(count), buffer))
        items = ctypes.cast(buffer, ctypes.POINTER(_Item))
        return [(items[i].name, items[i].formatted.value) for i in range(count.value)
                if items[i].formatted.status in (0, 1)]

    def close(self):
        if self.query:
            self.library.PdhCloseQuery(self.query)
            self.query = wintypes.HANDLE()


class WindowsGpuMetrics:
    """Coleta em segundo plano; consultas da API apenas leem a última amostra."""

    def __init__(self, interval=1.0, query_factory=PdhGpuQuery, adapter_provider=adapters, clock=time.monotonic, retry_interval=5.0):
        self.interval, self.query_factory, self.adapter_provider, self.clock = interval, query_factory, adapter_provider, clock
        self.retry_interval = retry_interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.target = None
        self.thread = None
        self.updated = 0.0
        self.state = self._state(None, 'warming_up')

    @staticmethod
    def _state(value, status, error=None):
        return {'gpu_utilization_percent': value, 'gpu_utilization_source': 'windows_wddm',
                'gpu_utilization_scope': 'selected_adapter_busiest_engine', 'gpu_utilization_status': status,
                'gpu_utilization_error': error}

    def sample(self, device_name: str, vendor: str) -> dict:
        if os.name != 'nt':
            return self._state(None, 'unavailable', 'Contadores WDDM disponíveis somente no Windows.')
        with self.lock:
            target = (device_name, vendor)
            if target != self.target:
                self.target = target
                self.state = self._state(None, 'warming_up')
                self.updated = self.clock()
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='gpu-metrics', daemon=True)
                self.thread.start()
            if self.clock() - self.updated > max(3.5, self.interval * 3):
                return self._state(None, 'warming_up', 'Aguardando uma nova amostra de GPU.')
            return dict(self.state)

    def _publish(self, target, state):
        with self.lock:
            if target == self.target:
                self.state, self.updated = state, self.clock()

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                target = self.target
            query = None
            try:
                luid = select_adapter(self.adapter_provider(), *target)
                query = self.query_factory()
                while not self.stop.wait(self.interval):
                    with self.lock:
                        if self.target != target:
                            break
                    value = busiest_engine(query.read(), luid)
                    self._publish(target, self._state(value, 'available' if value is not None else 'warming_up'))
            except Exception as error:
                self._publish(target, self._state(None, 'unavailable', str(error)))
                self.stop.wait(self.retry_interval)
            finally:
                if query is not None:
                    query.close()

    def close(self):
        self.stop.set()
