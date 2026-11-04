#!/usr/bin/env python3
"""check_t3.py — automated checker for Assignment 3 (CSSO 2026–2027).

Checks the outputs produced by t3.exe:
    C:\\Faculty\\CSSO\\T3\\processes.tsv                  (t3.exe ps)
    C:\\Faculty\\CSSO\\T3\\modules_<pid>.tsv, threads_<pid>.tsv   (t3.exe details <pid>)
    C:\\Faculty\\CSSO\\T3\\run\\results.tsv               (t3.exe run test_jobs.ini)
    C:\\Faculty\\CSSO\\T3\\run\\job_<name>.stdout.txt / .stderr.txt

Usage:
    python check_t3.py [--dir C:\\Faculty\\CSSO\\T3] [--ini test_jobs.ini] [--only ps|details|run]

Python standard library only (>= 3.10). It does not print points: the test codes (T3.xy)
correspond to the marking-scheme items; the assessor also checks at the presentation what cannot be tested automatically.
"""
from __future__ import annotations

import argparse
import configparser
import ctypes
import ctypes.wintypes as wt
import re
import sys
from pathlib import Path

RES = {"ok": 0, "fail": 0}
RESULT_COLUMNS = ["job", "command", "reason", "exit_code", "start_ms", "real_time_ms",
                  "cpu_time_ms", "peak_memory_kb", "total_processes", "stdout_bytes", "stderr_bytes", "truncated"]
PS_COLUMNS = ["pid", "ppid", "name", "threads", "session", "created_utc", "image", "user",
              "integrity", "elevated", "parent", "status"]
CAP = 1024 * 1024
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z$")


def ok(code: str, text: str) -> None:
    RES["ok"] += 1
    print(f"[OK] {code} {text}")


def fail(code: str, text: str, detail: str = "") -> None:
    RES["fail"] += 1
    print(f"[FAIL] {code} {text}" + (f" — {detail}" if detail else ""))


def info(text: str) -> None:
    print(f"[INFO] {text}")


def test(code: str, text: str, condition: bool, detail: str = "") -> bool:
    (ok if condition else fail)(code, text, *([] if condition else [detail]))
    return condition


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]] | None:
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as e:
        fail("T3.0", f"{path.name} is not valid UTF-8", str(e))
        return None
    if text.startswith("\ufeff"):
        info(f"{path.name} starts with a BOM; the specification requires UTF-8 without BOM")
        text = text[1:]
    lines = [l for l in text.replace("\r\n", "\n").split("\n") if l != ""]
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        values = l.split("\t")
        rows.append(dict(zip(header, values + ["<missing>"] * (len(header) - len(values)))))
    return header, rows


def number(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- live processes (ctypes)
class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def live_processes() -> list[tuple[int, str]]:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    res = []
    if snap in (None, wt.HANDLE(-1).value):
        return res
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(pe)
    ok_ = k32.Process32FirstW(snap, ctypes.byref(pe))
    while ok_:
        res.append((pe.th32ProcessID, pe.szExeFile))
        ok_ = k32.Process32NextW(snap, ctypes.byref(pe))
    k32.CloseHandle(snap)
    return res


def creation_time(pid: int) -> float | None:
    """The process creation time (Unix seconds), or None if it cannot be determined."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wt.HANDLE
    h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    c, e, k, u = wt.FILETIME(), wt.FILETIME(), wt.FILETIME(), wt.FILETIME()
    ok_ = k32.GetProcessTimes(wt.HANDLE(h), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
    k32.CloseHandle(wt.HANDLE(h))
    if not ok_:
        return None
    ft = (c.dwHighDateTime << 32) | c.dwLowDateTime
    return (ft - 116444736000000000) / 1e7


# ---------------------------------------------------------------- ps
def check_ps(d: Path) -> None:
    path = d / "processes.tsv"
    t = read_tsv(path)
    if t is None:
        fail("T3.1a", "processes.tsv exists", f"missing {path} (run: t3.exe ps)")
        return
    header, r = t
    test("T3.1a", "processes.tsv: the header has exactly the required columns", header == PS_COLUMNS,
         f"found: {header}")
    pids = [number(x.get("pid")) for x in r]
    test("T3.1a", "processes.tsv: at least 20 processes, numeric and unique PIDs",
         len(r) >= 20 and None not in pids and len(set(pids)) == len(pids), f"{len(r)} rows")
    test("T3.1a", "processes.tsv: the System process (PID 4) appears", 4 in pids)
    statuses = {"OK", "PARTIAL", "ACCESS_DENIED", "TERMINATED", "SYSTEM"}
    invalid = [x for x in r if x.get("status") not in statuses and not str(x.get("status", "")).startswith("ERROR:")]
    test("T3.1d", "the status column contains only permitted values", not invalid,
         f"e.g.: {[x.get('status') for x in invalid[:3]]}")
    okr = [x for x in r if x.get("status") == "OK"]
    test("T3.1b", "there are processes with status OK (at least 10)", len(okr) >= 10, f"{len(okr)}")
    good_img = [x for x in okr if re.match(r"^[A-Za-z]:\\", x.get("image", ""))]
    test("T3.1b", "the OK processes have the full image path (QueryFullProcessImageNameW)",
         okr and len(good_img) >= 0.95 * len(okr), f"{len(good_img)}/{len(okr)}")
    matching = [x for x in good_img if x["image"].lower().endswith("\\" + x["name"].lower())]
    test("T3.1b", "the snapshot name matches the image (≥ 95%)",
         good_img and len(matching) >= 0.95 * len(good_img), f"{len(matching)}/{len(good_img)}")
    iso_ok = [x for x in okr if ISO.match(x.get("created_utc", ""))]
    test("T3.1b", "created_utc in ISO 8601 UTC format for the OK processes",
         okr and len(iso_ok) == len(okr), f"{len(iso_ok)}/{len(okr)}")
    tok_ok = [x for x in okr if "\\" in x.get("user", "") and
              x.get("integrity") in {"Low", "Medium", "High", "System", "Untrusted"} and x.get("elevated") in {"yes", "no"}]
    test("T3.1b", "user (DOMAIN\\name), integrity and elevated filled in for the OK processes",
         okr and len(tok_ok) == len(okr), f"{len(tok_ok)}/{len(okr)}")
    denied = [x for x in r if x.get("status") == "ACCESS_DENIED"]
    if denied:
        test("T3.1d", "the processes with ACCESS_DENIED have no invented data (image = -)",
             all(x.get("image") == "-" for x in denied))
    else:
        info("no ACCESS_DENIED process (run t3.exe ps non-elevated to see how the denial is handled)")
    # we recompute the parent status from the file
    by_pid = {number(x["pid"]): x for x in r}
    wrong = []
    for x in r:
        pid, ppid = number(x["pid"]), number(x["ppid"])
        if pid == 0 or ppid == pid:
            expected = {"-"}
        elif ppid not in by_pid:
            expected = {"nonexistent"}
        else:
            p = by_pid[ppid]
            if ISO.match(p.get("created_utc", "")) and ISO.match(x.get("created_utc", "")) and p["created_utc"] > x["created_utc"]:
                expected = {"pid_reused"}
            elif ISO.match(p.get("created_utc", "")) and ISO.match(x.get("created_utc", "")) and p["created_utc"] == x["created_utc"]:
                expected = {"active", "pid_reused"}      # the same second: the order cannot be decided from the text
            else:
                expected = {"active"}
        if x.get("parent") not in expected:
            wrong.append((x["pid"], x.get("parent"), "/".join(sorted(expected))))
    test("T3.1c", "the parent column (active / nonexistent / pid_reused) is consistent with the PIDs and times",
         not wrong, f"{len(wrong)} wrong, e.g.: {wrong[:3]}")


# ---------------------------------------------------------------- details
def check_details(d: Path) -> None:
    modules = sorted(d.glob("modules_*.tsv"))
    if not modules:
        fail("T3.2a", "modules_<pid>.tsv exists", "run: t3.exe details <pid> (e.g. the PID of one of your own processes)")
        return
    for m in modules:
        pid = m.stem.split("_", 1)[1]
        t = read_tsv(m)
        if t is None:
            continue
        header, r = t
        test("T3.2a", f"{m.name}: header base/size/name/path", header == ["base", "size", "name", "path"], str(header))
        names = [x.get("name", "").lower() for x in r]
        test("T3.2a", f"{m.name}: contains the .exe image and ntdll.dll",
             bool(r) and names[0].endswith(".exe") and "ntdll.dll" in names, f"first: {names[:3]}")
        test("T3.2a", f"{m.name}: base addresses in hexadecimal, numeric sizes",
             all(re.match(r"^(0x)?[0-9A-Fa-f]+$", x.get("base", "")) and number(x.get("size")) for x in r))
        f = d / f"threads_{pid}.tsv"
        tf = read_tsv(f)
        if tf is None:
            fail("T3.2b", f"threads_{pid}.tsv exists")
            continue
        header, r = tf
        test("T3.2b", f"threads_{pid}.tsv: header tid/base_priority/description", header == ["tid", "base_priority", "description"], str(header))
        test("T3.2b", f"threads_{pid}.tsv: at least one thread, unique numeric TIDs",
             bool(r) and all(number(x.get("tid")) for x in r) and len({x['tid'] for x in r}) == len(r))


# ---------------------------------------------------------------- run
def read_ini(ini: Path) -> list[str]:
    raw = ini.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    cp.read_string(raw.decode(enc))
    return [s for s in cp.sections() if s.lower() != "general"]


def content(d: Path, job: str, stream: str) -> bytes | None:
    try:
        return (d / f"job_{job}.{stream}.txt").read_bytes()
    except FileNotFoundError:
        return None


def check_run(d: Path, ini: Path) -> None:
    run = d / "run"
    t = read_tsv(run / "results.tsv")
    if t is None:
        fail("T3.3a", "run\\results.tsv exists", "run: t3.exe run test_jobs.ini")
        return
    header, r = t
    test("T3.3a", "results.tsv: the header has exactly the required columns", header == RESULT_COLUMNS, f"found: {header}")
    jobs = read_ini(ini)
    names = [x.get("job") for x in r]
    test("T3.3a", "results.tsv: one row for each job, in the INI order", names == jobs,
         f"expected {jobs}, found {names}")
    J = {x.get("job"): x for x in r}

    def row(job: str) -> dict[str, str]:
        return J.get(job, {})

    def check_reason(code: str, job: str, reason: str, exit_code: str | None = None) -> None:
        x = row(job)
        cond = x.get("reason") == reason and (exit_code is None or x.get("exit_code", "").upper() == exit_code.upper())
        test(code, f"[{job}] reason {reason}" + (f", code {exit_code}" if exit_code else ""), cond,
             f"found reason={x.get('reason')} code={x.get('exit_code')}")

    # T3.3 — launch, pipes, arguments
    check_reason("T3.3b", "simple_ok", "OK", "0x00000000")
    check_reason("T3.3b", "failed_launch", "LAUNCH_ERROR:2")
    test("T3.3b", "[failed_launch] the measurement fields are '-' (the process did not exist)",
         row("failed_launch").get("exit_code") == "-" and row("failed_launch").get("real_time_ms") == "-")
    out = content(run, "arguments", "stdout")
    expected = "argc=3\n[1] Anna has apples\n[2] café, naïve Ω\n[3] final\n".encode("utf-8")
    test("T3.3c", "[arguments] the arguments with spaces and diacritics reach the child intact", out == expected,
         f"received {out[:120]!r}" if out is not None else "job_arguments.stdout.txt is missing")
    out = content(run, "directory", "stdout")
    test("T3.3c", "[directory] the child's current directory is C:\\Windows",
         out is not None and out.decode("utf-8", "replace").strip().lower() == "c:\\windows", f"received {out!r}")
    x = row("flood")
    so, se = content(run, "flood", "stdout"), content(run, "flood", "stderr")
    test("T3.3d", "[flood] 3 MiB on each stream, without blocking (reason OK)", x.get("reason") == "OK",
         f"reason={x.get('reason')}")
    test("T3.3d", "[flood] stdout_bytes = stderr_bytes = 3145728 (total received)",
         number(x.get("stdout_bytes")) == 3 * CAP and number(x.get("stderr_bytes")) == 3 * CAP,
         f"{x.get('stdout_bytes')} / {x.get('stderr_bytes')}")
    test("T3.3d", "[flood] the capture files have exactly 1 MiB and truncated = both",
         so is not None and se is not None and len(so) == CAP and len(se) == CAP and x.get("truncated") == "both",
         f"{None if so is None else len(so)} / {None if se is None else len(se)}, truncated={x.get('truncated')}")
    test("T3.3d", "[simple_ok] truncated = no", row("simple_ok").get("truncated") == "no")

    # T3.4 — Job Object, limits, reasons
    check_reason("T3.4a", "nonzero_code", "NONZERO_CODE", "0x00000007")
    check_reason("T3.4b", "timeout", "TIMEOUT")
    tr = number(row("timeout").get("real_time_ms"))
    test("T3.4b", "[timeout] the real duration is between 1800 and 6000 ms (the limit is 2000 ms)",
         tr is not None and 1800 <= tr <= 6000, f"{tr}")
    check_reason("T3.4c", "descendants", "TIMEOUT")
    x = row("descendants")
    tr, pr = number(x.get("real_time_ms")), number(x.get("total_processes"))
    test("T3.4c", "[descendants] the job ends although the descendants keep the pipes open (≤ 8000 ms)",
         tr is not None and tr <= 8000, f"{tr}")
    test("T3.4c", "[descendants] total_processes ≥ 4 (the child and its 3 descendants)", pr is not None and pr >= 4, f"{pr}")
    # Only the processes created during the run count (the 10-minute window before results.tsv
    # was written). A t3_child started manually beforehand (for example for `t3.exe details`) is ignored.
    try:
        run_end = (run / "results.tsv").stat().st_mtime
    except OSError:
        run_end = None
    alive, ignored = [], []
    for pid, name in live_processes():
        if name.lower() != "t3_child.exe":
            continue
        t0 = creation_time(pid)
        if run_end is not None and t0 is not None and not (run_end - 600 <= t0 <= run_end + 1):
            ignored.append(pid)
        else:
            alive.append(pid)
    if ignored:
        info(f"T3.4c: ignoring t3_child.exe processes started outside the run (probably manually): PID {ignored}")
    test("T3.4c", "no t3_child.exe created during the run is still running after the jobs have ended", not alive,
         f"PIDs still running: {alive}")
    check_reason("T3.4d", "memory", "MEMORY_LIMIT")
    v = number(row("memory").get("peak_memory_kb"))
    test("T3.4d", "[memory] peak_memory_kb close to the 64 MiB limit (45000–67000 KiB)",
         v is not None and 45000 <= v <= 67000, f"{v}")
    check_reason("T3.4e", "processes", "PROCESS_LIMIT")
    pr = number(row("processes").get("total_processes"))
    test("T3.4e", "[processes] total_processes between 3 and 4 (limit 3; the refused process may be counted)",
         pr is not None and 3 <= pr <= 4, f"{pr}")
    check_reason("T3.4f", "crash", "CRASH:0xC0000005", "0xC0000005")

    # T3.5 — accounting
    launched = [x for x in r if not str(x.get("reason", "")).startswith("LAUNCH_ERROR")]
    fields = ["start_ms", "real_time_ms", "cpu_time_ms", "peak_memory_kb", "total_processes",
              "stdout_bytes", "stderr_bytes"]
    bad = [x.get("job") for x in launched if any(number(x.get(c)) is None for c in fields)
           or not re.match(r"^0x[0-9A-Fa-f]{8}$", x.get("exit_code", ""))]
    test("T3.5a", "the accounting fields are numeric, exit_code in the form 0xXXXXXXXX", not bad, f"jobs: {bad}")
    fl = row("flood")
    test("T3.5a", "[flood] cpu_time_ms ≥ 0 and total_processes ≥ 1",
         (number(fl.get("cpu_time_ms")) or 0) >= 0 and (number(fl.get("total_processes")) or 0) >= 1)
    sl = number(row("sleep_1500").get("real_time_ms"))
    test("T3.5a", "[sleep_1500] the real duration of a \"sleep 1500\" is between 1400 and 4000 ms",
         sl is not None and 1400 <= sl <= 4000, f"{sl}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="T3 automated checker")
    ap.add_argument("--dir", default=r"C:\Faculty\CSSO\T3")
    ap.add_argument("--ini", default=str(Path(__file__).with_name("test_jobs.ini")))
    ap.add_argument("--only", choices=["ps", "details", "run"])
    a = ap.parse_args()
    d = Path(a.dir)
    if a.only in (None, "ps"):
        check_ps(d)
    if a.only in (None, "details"):
        check_details(d)
    if a.only in (None, "run"):
        check_run(d, Path(a.ini))
    total = RES["ok"] + RES["fail"]
    print(f"Summary: {RES['ok']}/{total} automated tests passed")
    return 0 if RES["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
