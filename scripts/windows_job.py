"""Mantém os subprocessos de uma validação no mesmo grupo do Windows."""
import ctypes
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [('ProcessTime', ctypes.c_longlong), ('JobTime', ctypes.c_longlong),
                ('Flags', wintypes.DWORD), ('MinimumWorkingSet', ctypes.c_size_t),
                ('MaximumWorkingSet', ctypes.c_size_t), ('ActiveProcesses', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('Priority', wintypes.DWORD), ('Scheduling', wintypes.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('Basic', BasicLimits), ('IoCounters', ctypes.c_ulonglong * 6),
                ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]


class WindowsJob:
    def __init__(self, child):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.api.AssignProcessToJobObject(self.handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def terminate(self):
        if self.handle and not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, child):
        # O processo nasce suspenso para que nenhum filho escape do grupo
        # antes da associação, inclusive o launcher de ambientes virtuais.
        native = ctypes.WinDLL('ntdll')
        native.NtResumeProcess.argtypes = [wintypes.HANDLE]
        native.NtResumeProcess.restype = ctypes.c_long
        status = native.NtResumeProcess(int(child._handle))
        if status != 0:
            raise OSError(f'Não foi possível retomar a tarefa supervisionada (NTSTATUS {status:#x}).')

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
