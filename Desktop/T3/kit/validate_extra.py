"""Supplemental lifecycle/configuration checks; does not replace course tests."""
import ctypes as c
from ctypes import wintypes as w
import hashlib, os, subprocess, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
exe, directory = Path(sys.argv[1]), Path(sys.argv[2])
base = directory / "extra"
base.mkdir(exist_ok=True)
env = dict(os.environ, T3_VALIDATION_ROOT=str(base))
child = directory / "t3_child.exe"
def run(args): return subprocess.run([str(exe), *args], env=env, capture_output=True, timeout=10)
def ini(name, text):
    f = base / name; f.write_text(text, encoding="utf-16"); return str(f)
def check(description, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {description}")
    if not condition: raise AssertionError(description)
marker = base / "must_not_exist.txt"
bad = ini("invalid.ini", f'[would_run]\ncommand=""{child}" write_file "{marker}""\n[invalid]\ntimeout_ms=10\n')
r = run(["run", bad])
check("invalid later job rejects entire INI before launching first job", r.returncode == 2 and not marker.exists())
for args in [["details","-1"],["experiment","threads","15","64","0"],
             ["experiment","threads","128","4097","0"],["experiment","threads","128","64","2"],
             ["run",str(base/"missing.ini")]]:
    check("invalid CLI/config: " + " ".join(args[:2]), run(args).returncode == 2)
unicode_dir = base / "spațiu Ω"
unicode_dir.mkdir(exist_ok=True)
custom = ini("unicode.ini", f'[unicode]\ncommand=""{child}" directory"\ndirectory={unicode_dir}\n')
check("Unicode working directory launches", run(["run",custom]).returncode == 0)
check("Unicode working directory arrives intact", (base/"run"/"job_unicode.stdout.txt").read_text(encoding="utf-8").strip() == str(unicode_dir))

class PE(c.Structure):
    _fields_=[("dwSize",w.DWORD),("cntUsage",w.DWORD),("pid",w.DWORD),
              ("heap",c.c_size_t),("module",w.DWORD),("threads",w.DWORD),
              ("ppid",w.DWORD),("priority",w.LONG),("flags",w.DWORD),("name",w.WCHAR*260)]
k=c.WinDLL("kernel32",use_last_error=True)
k.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD]; k.CreateToolhelp32Snapshot.restype=w.HANDLE
k.Process32FirstW.argtypes=[w.HANDLE,c.POINTER(PE)];k.Process32NextW.argtypes=[w.HANDLE,c.POINTER(PE)]
k.CloseHandle.argtypes=[w.HANDLE]
def children():
    h=k.CreateToolhelp32Snapshot(2,0); p=PE();p.dwSize=c.sizeof(p); found=set()
    try:
        ok=k.Process32FirstW(h,c.byref(p))
        while ok:
            if p.name.lower()=="t3_child.exe": found.add(p.pid)
            ok=k.Process32NextW(h,c.byref(p))
    finally: k.CloseHandle(h)
    return found
before=children()
kill=ini("kill_on_close.ini",f'[long]\ncommand=""{child}" children 3 30000"\ntimeout_ms=60000\n')
p=subprocess.Popen([str(exe),"run",kill],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
created=set()
try:
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        created=children()-before
        if len(created)>=3: break
        time.sleep(.05)
    check("long job created descendants before supervisor termination",len(created)>=3)
    p.kill();p.communicate(timeout=5)
    deadline=time.monotonic()+5
    while children() & created and time.monotonic()<deadline: time.sleep(.05)
    check("KILL_ON_JOB_CLOSE terminates descendants when supervisor is killed",not(children()&created))
finally:
    if p.poll() is None: p.kill();p.communicate(timeout=5)
print("Supplemental checks completed successfully.")
