"""Run the unchanged evaluator and record this process's Windows peak working set."""
import ctypes
import json
from pathlib import Path
import runpy
import sys

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
             'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
             'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_working_set_bytes': counters.PeakWorkingSetSize,
            'peak_commit_bytes': counters.PeakPagefileUsage,
            'measurement': 'Windows GetProcessMemoryInfo in the evaluator process'}


def main():
    if '--output' not in sys.argv:
        raise ValueError('Supply an explicit --output path for resource recording.')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    # The original argparse, loading, fixed windows and scorer execute unchanged.
    runpy.run_path(str(CODE_ROOT / 'evaluate.py'), run_name='__main__')
    resources = peak_memory()
    output.with_suffix('.resources.json').write_text(json.dumps(resources, indent=2) + '\n')
    print(json.dumps({'resources': resources}), flush=True)


if __name__ == '__main__':
    main()
