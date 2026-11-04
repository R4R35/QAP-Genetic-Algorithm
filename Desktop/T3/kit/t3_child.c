/* t3_child.c — test program for Assignment 3 (CSSO 2026–2027).
 *
 * Build (Developer Command Prompt for Visual Studio):
 *     cl /W4 /O2 /DUNICODE /D_UNICODE t3_child.c
 * Copy t3_child.exe to C:\Faculty\CSSO\T3\ (test_jobs.ini looks for it there).
 *
 * Modes (first argument):
 *     exit <code>               exits with the given code (decimal)
 *     args [a1 a2 ...]          writes "argc=N" to stdout, then "[i] <argument>" on separate lines
 *     directory                 writes the current directory to stdout
 *     env <VARIABLE>            writes the environment variable's value to stdout (or "<missing>")
 *     integrity                 writes the integrity level to stdout: Low / Medium / High / System
 *     flood <MiB>               writes <MiB> MiB to stdout AND <MiB> MiB to stderr, alternately, in 4 KiB blocks
 *     sleep <ms>                sleeps <ms> milliseconds, then exits with 0
 *     memory <MiB>              allocates and touches memory 1 MiB at a time; on failure writes to stderr and exits with 3
 *     children <n> <ms>         launches n children "t3_child sleep <ms>" that INHERIT stdout/stderr,
 *                               then exits immediately with 0 (if a launch fails: stderr + code 4)
 *     crash                     accesses the NULL address -> unhandled exception 0xC0000005
 *     write_file <path>         creates <path> and writes "test"; exits with 0 or with the error's Win32 code
 *     threads <stack_kib> <reservation 0|1> <max>
 *                               (experiment E1) creates threads that wait for an event, with
 *                               CreateThread(dwStackSize = stack_kib KiB, flag STACK_SIZE_PARAM_IS_A_RESERVATION
 *                               if reservation = 1), until the first failure or until <max> threads; writes to stdout
 *                               "THREADS <n> CODE <code> PRIVATE_BEFORE <bytes> PRIVATE_AFTER <bytes>" (CODE 0 = cap
 *                               reached), then releases the threads, writes "CLOSED <n>" and exits with 0.
 *                               Run it ONLY in a Job Object with a memory limit (see Assignment 3, E1).
 *
 * All text is written in UTF-8 through WriteFile (without CRT translation), with a "\n" terminator.
 */
#include <windows.h>
#include <psapi.h>
#include <stdlib.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "advapi32.lib")   /* OpenProcessToken, GetTokenInformation (integrity mode) */
#include <string.h>
#include <wchar.h>

static void write_text(HANDLE h, const wchar_t *text)
{
    char buf[8192];
    int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, buf, sizeof(buf), NULL, NULL);
    if (n > 1) {
        DWORD written;
        WriteFile(h, buf, (DWORD)(n - 1), &written, NULL);
    }
}

static void write_line(HANDLE h, const wchar_t *text)
{
    write_text(h, text);
    write_text(h, L"\n");
}

static int mode_args(int argc, wchar_t **argv)
{
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    wchar_t line[4096];
    swprintf(line, 4096, L"argc=%d", argc - 2);
    write_line(out, line);
    for (int i = 2; i < argc; i++) {
        swprintf(line, 4096, L"[%d] %ls", i - 1, argv[i]);
        write_line(out, line);
    }
    return 0;
}

static int mode_integrity(void)
{
    HANDLE tok;
    const wchar_t *name = L"Unknown";
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) {
        BYTE buf[256];
        DWORD size;
        if (GetTokenInformation(tok, TokenIntegrityLevel, buf, sizeof(buf), &size)) {
            PSID sid = ((TOKEN_MANDATORY_LABEL *)buf)->Label.Sid;
            DWORD rid = *GetSidSubAuthority(sid, *GetSidSubAuthorityCount(sid) - 1);
            name = rid >= SECURITY_MANDATORY_SYSTEM_RID ? L"System" :
                   rid >= SECURITY_MANDATORY_HIGH_RID ? L"High" :
                   rid >= SECURITY_MANDATORY_MEDIUM_RID ? L"Medium" :
                   rid >= SECURITY_MANDATORY_LOW_RID ? L"Low" : L"Untrusted";
        }
        CloseHandle(tok);
    }
    write_line(GetStdHandle(STD_OUTPUT_HANDLE), name);
    return 0;
}

static int mode_flood(unsigned mib)
{
    static char block[4096];
    memset(block, 'x', sizeof(block));
    block[sizeof(block) - 1] = '\n';
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE), err = GetStdHandle(STD_ERROR_HANDLE);
    unsigned blocks = mib * 256;                  /* 256 × 4 KiB = 1 MiB */
    for (unsigned i = 0; i < blocks; i++) {
        DWORD written;
        if (!WriteFile(out, block, sizeof(block), &written, NULL)) return 5;
        if (!WriteFile(err, block, sizeof(block), &written, NULL)) return 5;
    }
    return 0;
}

static int mode_memory(unsigned mib)
{
    for (unsigned i = 0; i < mib; i++) {
        char *p = (char *)VirtualAlloc(NULL, 1u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (p == NULL) {
            wchar_t line[128];
            swprintf(line, 128, L"allocation failed after %u MiB (error %lu)", i, GetLastError());
            write_line(GetStdHandle(STD_ERROR_HANDLE), line);
            return 3;
        }
        memset(p, 1, 1u << 20);                   /* the memory is touched, not just reserved */
    }
    return 0;
}

static int mode_children(wchar_t *self, unsigned n, unsigned ms)
{
    for (unsigned i = 0; i < n; i++) {
        wchar_t line[MAX_PATH + 64];
        swprintf(line, ARRAYSIZE(line), L"\"%ls\" sleep %u", self, ms);
        STARTUPINFOW si;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION pi;
        /* bInheritHandles = TRUE: the descendants keep the pipe ends open */
        if (!CreateProcessW(NULL, line, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
            wchar_t msg[128];
            swprintf(msg, 128, L"launching child %u failed: %lu", i + 1, GetLastError());
            write_line(GetStdHandle(STD_ERROR_HANDLE), msg);
            return 4;
        }
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    return 0;
}

static DWORD WINAPI thread_wait(LPVOID p)
{
    WaitForSingleObject((HANDLE)p, INFINITE);
    return 0;
}

static SIZE_T private_memory(void)
{
    PROCESS_MEMORY_COUNTERS_EX pm;
    ZeroMemory(&pm, sizeof(pm));
    pm.cb = sizeof(pm);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm, sizeof(pm))) return 0;
    return pm.PrivateUsage;
}

static int mode_threads(unsigned stack_kib, unsigned reservation, unsigned max)
{
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (max == 0 || max > 10000) {                 /* safety cap */
        write_line(GetStdHandle(STD_ERROR_HANDLE), L"threads: <max> must be between 1 and 10000");
        return 2;
    }
    HANDLE stop = CreateEventW(NULL, TRUE, FALSE, NULL);
    HANDLE *threads = (HANDLE *)calloc(max, sizeof(HANDLE));
    if (stop == NULL || threads == NULL) {
        write_line(GetStdHandle(STD_ERROR_HANDLE), L"threads: the initial resources could not be allocated");
        if (stop) CloseHandle(stop);
        free(threads);
        return 3;
    }
    SIZE_T before = private_memory();
    unsigned n = 0;
    DWORD code = 0;
    for (; n < max; n++) {
        threads[n] = CreateThread(NULL, (SIZE_T)stack_kib * 1024, thread_wait, stop,
                                  reservation ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0, NULL);
        if (threads[n] == NULL) {
            code = GetLastError();
            break;
        }
    }
    SIZE_T after = private_memory();
    wchar_t line[256];
    swprintf(line, ARRAYSIZE(line), L"THREADS %u CODE %lu PRIVATE_BEFORE %llu PRIVATE_AFTER %llu",
             n, code, (unsigned long long)before, (unsigned long long)after);
    write_line(out, line);
    SetEvent(stop);                                 /* all threads can finish */
    for (unsigned i = 0; i < n; i++) {
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
    }
    CloseHandle(stop);
    free(threads);
    swprintf(line, ARRAYSIZE(line), L"CLOSED %u", n);
    write_line(out, line);
    return 0;
}

static int mode_write_file(const wchar_t *path)
{
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        wchar_t msg[MAX_PATH + 64];
        swprintf(msg, ARRAYSIZE(msg), L"CreateFileW(%ls): error %lu", path, e);
        write_line(GetStdHandle(STD_ERROR_HANDLE), msg);
        return (int)e;
    }
    DWORD written;
    WriteFile(h, "test", 4, &written, NULL);
    CloseHandle(h);
    return 0;
}

int wmain(int argc, wchar_t **argv)
{
    /* No error windows (WER / "Insert disk"): the tests must not block. */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    if (argc < 2) {
        write_line(GetStdHandle(STD_ERROR_HANDLE), L"usage: t3_child <mode> [arguments] (see the source)");
        return 2;
    }
    const wchar_t *mode = argv[1];
    unsigned a2 = argc > 2 ? (unsigned)wcstoul(argv[2], NULL, 10) : 0;
    unsigned a3 = argc > 3 ? (unsigned)wcstoul(argv[3], NULL, 10) : 0;

    if (!wcscmp(mode, L"exit"))       return (int)a2;
    if (!wcscmp(mode, L"args"))       return mode_args(argc, argv);
    if (!wcscmp(mode, L"integrity"))  return mode_integrity();
    if (!wcscmp(mode, L"flood"))      return mode_flood(a2);
    if (!wcscmp(mode, L"sleep"))      { Sleep(a2); return 0; }
    if (!wcscmp(mode, L"memory"))     return mode_memory(a2);
    if (!wcscmp(mode, L"children"))   return mode_children(argv[0], a2, a3);
    if (!wcscmp(mode, L"write_file") && argc > 2) return mode_write_file(argv[2]);
    if (!wcscmp(mode, L"threads") && argc > 4)
        return mode_threads(a2, a3, (unsigned)wcstoul(argv[4], NULL, 10));
    if (!wcscmp(mode, L"directory")) {
        wchar_t d[MAX_PATH];
        GetCurrentDirectoryW(MAX_PATH, d);
        write_line(GetStdHandle(STD_OUTPUT_HANDLE), d);
        return 0;
    }
    if (!wcscmp(mode, L"env") && argc > 2) {
        wchar_t v[4096];
        DWORD n = GetEnvironmentVariableW(argv[2], v, 4096);
        write_line(GetStdHandle(STD_OUTPUT_HANDLE), (n > 0 && n < 4096) ? v : L"<missing>");
        return 0;
    }
    if (!wcscmp(mode, L"crash")) {
        volatile int *p = NULL;
        *p = 42;                                   /* 0xC0000005 STATUS_ACCESS_VIOLATION */
        return 0;
    }
    write_line(GetStdHandle(STD_ERROR_HANDLE), L"unknown mode");
    return 2;
}
