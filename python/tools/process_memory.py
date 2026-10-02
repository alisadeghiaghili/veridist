"""Process memory facts shared by the evidence collectors.

``peak_rss_bytes`` is the high-water mark of the process resident set, so it is
monotonic within a process: the growth observed across a measured region is
zero whenever an earlier region already reached the same peak.
"""

from __future__ import annotations

import ctypes
import sys


class _ProcessMemoryCounters(ctypes.Structure):
    """``PROCESS_MEMORY_COUNTERS_EX`` from ``psapi.h``."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def peak_rss_bytes() -> int:
    """Return the peak resident set of this process, or fail rather than guess.

    Windows reads ``PeakWorkingSetSize`` through ``GetProcessMemoryInfo``. POSIX
    reads ``ru_maxrss``, which the kernel reports in bytes on macOS and in KiB
    elsewhere.
    """

    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        getter = ctypes.windll.psapi.GetProcessMemoryInfo
        getter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        getter.restype = ctypes.c_int
        if not getter(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise RuntimeError("cannot obtain Windows process RSS")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)
