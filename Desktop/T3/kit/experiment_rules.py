"""CSSO 2026–2027 · Assignment 3 · the automated rules for experiment E1.

Run (from Kit/):  python check_experiments.py T3 --md <path>\\EXPERIMENTS.md
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from datetime import datetime, timezone
from pathlib import Path

import experiment_common as ec

DEFAULT_DIR = r"C:\Faculty\CSSO\T3"
EXPERIMENTS = {
    "E1": "How many threads fit in a memory-limited job",
}
E1_COMBINATIONS = [("64", "0"), ("64", "1"), ("1024", "0"), ("1024", "1")]
LIMIT_CODES = {8, 1455}             # ERROR_NOT_ENOUGH_MEMORY, ERROR_COMMITMENT_LIMIT
INIT_TOLERANCE = 16                 # handles opened only once by the system on first use


# ---------------------------------------------------------------- remaining t3_child processes
class _PE32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def _live_children() -> list[tuple[int, float | None]]:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    k32.OpenProcess.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    res: list[tuple[int, float | None]] = []
    if snap in (None, wt.HANDLE(-1).value):
        return res
    pe = _PE32()
    pe.dwSize = ctypes.sizeof(pe)
    ok = k32.Process32FirstW(wt.HANDLE(snap), ctypes.byref(pe))
    while ok:
        if pe.szExeFile.lower() == "t3_child.exe":
            t0 = None
            h = k32.OpenProcess(0x1000, False, pe.th32ProcessID)
            if h:
                c, e, k, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
                if k32.GetProcessTimes(wt.HANDLE(h), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                    t0 = (((c.dwHighDateTime << 32) | c.dwLowDateTime) - 116444736000000000) / 1e7
                k32.CloseHandle(wt.HANDLE(h))
            res.append((pe.th32ProcessID, t0))
        ok = k32.Process32NextW(wt.HANDLE(snap), ctypes.byref(pe))
    k32.CloseHandle(wt.HANDLE(snap))
    return res


def _ts(run: str) -> float:
    return datetime.strptime(run, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _details(x: dict) -> dict[str, str]:
    return ec.params(x.get("details", ""))


# ---------------------------------------------------------------- checks
def check(r: ec.Report, rows: list[dict], d: Path) -> None:
    # ---------------- E1
    e1: dict[tuple[str, str], dict] = {}
    for kb, res in E1_COMBINATIONS:
        lines = ec.select(rows, "E1", memory_mb="128", stack_kb=kb, reservation=res)
        x = lines[-1] if lines else None
        mx = ec.integer(x["_param"].get("max")) if x else None
        ok = x is not None and x["unit"] == "threads" and (ec.integer(x["value"]) or 0) >= 1 \
            and mx is not None and 1 <= mx <= 10000
        r.test("T3.E1a", f"E1 memory_mb=128, stack_kb={kb}, reservation={res}: a line with the unit \"threads\" and the max parameter",
               ok, "missing" if x is None else f"value={x['value']} unit={x['unit']} max={mx}")
        if ok:
            e1[(kb, res)] = x
    if len(e1) == 4:
        n = {k: ec.integer(v["value"]) for k, v in e1.items()}
        c = {k: ec.integer(v["error_code"]) for k, v in e1.items()}
        mx = {k: ec.integer(v["_param"]["max"]) for k, v in e1.items()}
        ok0 = all(c[(kb, "0")] in LIMIT_CODES and n[(kb, "0")] < mx[(kb, "0")] for kb in ("64", "1024"))
        r.test("T3.E1b", "without reservation: the job limit stops creation (code 8 or 1455) before the cap", ok0,
               f"codes {c[('64', '0')]}, {c[('1024', '0')]}")
        r.test("T3.E1b", "stack_kb=1024 without reservation: 64–128 threads (≈ 128 MiB / 1 MiB)",
               64 <= n[("1024", "0")] <= 128, f"{n[('1024', '0')]} threads")
        r.test("T3.E1b", "stack_kb=64 without reservation: more than 4 times as many threads as with 1024 KiB, at most 2048",
               n[("64", "0")] > 4 * n[("1024", "0")] and n[("64", "0")] <= 2048, f"{n[('64', '0')]} threads")
        ok1 = all((c[(kb, "1")] == 0 and n[(kb, "1")] == mx[(kb, "1")]) or c[(kb, "1")] in LIMIT_CODES
                  for kb in ("64", "1024")) and all(n[(kb, "1")] >= n[(kb, "0")] for kb in ("64", "1024"))
        r.test("T3.E1b", "with reservation: at least as many threads as without (cap reached with code 0 or limit 8/1455)",
               ok1, f"{n[('64', '1')]}/{c[('64', '1')]}, {n[('1024', '1')]}/{c[('1024', '1')]}")
        det = {k: _details(v) for k, v in e1.items()}
        ok_det = all(ec.number(x.get("private_per_thread_kib")) is not None and ec.number(x.get("private_per_thread_kib")) > 0
                     and x.get("remaining_processes") == "0" and x.get("child_exit") for x in det.values())
        r.test("T3.E1c", "details: private_per_thread_kib > 0, child_exit, remaining_processes=0 on all 4 lines", ok_det,
               "; ".join(v.get("details", "") for v in e1.values())[:200])
        deltas = [ec.integer(v["handles_after"]) - ec.integer(v["handles_before"]) for v in e1.values()
                  if ec.integer(v["handles_after"]) is not None and ec.integer(v["handles_before"]) is not None]
        ok_h = len(deltas) == 4 and max(deltas) <= INIT_TOLERANCE and max(deltas) - min(deltas) <= 2
        r.test("T3.E1d", f"E1 cleanup: the handle difference is ≤ {INIT_TOLERANCE} and does not depend on the number of threads "
                         "(variation ≤ 2 between combinations)", ok_h, f"differences {deltas}")

    # ---------------- remaining processes
    times = [_ts(x["run"]) for x in rows if x["experiment"] == "E1"]
    if times:
        start = min(times) - 1
        alive = [pid for pid, t0 in _live_children() if t0 is None or t0 >= start]
        r.test("T3.E1d", "no t3_child.exe started after the experiments began is still running", not alive,
               f"PIDs: {alive} (first close the t3_child processes started manually, for example for `details`)")
