# CSSO T3 — Processes, threads and Job Objects

## Open and build
Open **t3.sln**, select **Release | x64**, then Build Solution.
The project GUID is preserved from your existing Visual Studio starter; all
four implementation files are included. Output: x64\Release\t3.exe.
CMakeLists.txt is also supplied.

This is a completed COPY of C:\Users\nitoi\source\repos\CSSO\t3.
The original folder is outside the session's writable workspace and was not
modified. Open this copy directly, or copy src/, kit/, t3.sln and
t3/t3.vcxproj into the corresponding locations of your existing solution.
The old Hello World t3.cpp is no longer included in the project.

From an x64 Native Tools Command Prompt for VS 2022, in this package:

    msbuild t3.sln /p:Configuration=Release /p:Platform=x64
    mkdir build
    cl /nologo /W4 /O2 /utf-8 /DUNICODE /D_UNICODE kit\t3_child.c /Fo:build\t3_child.obj /Fe:build\t3_child.exe
    mkdir C:\Faculty\CSSO\T3
    copy /y build\t3_child.exe C:\Faculty\CSSO\T3\
    x64\Release\t3.exe ps
    x64\Release\t3.exe details YOUR_PID
    x64\Release\t3.exe run kit\test_jobs.ini
    python kit\check_t3.py

Use a PID belonging to a live process you own. You can start
C:\Faculty\CSSO\T3\t3_child.exe sleep 60000 in another terminal; let it exit
before running the checker. ps may return 1 because protected processes have
incomplete information; it still writes the entire snapshot.

## Files
- src/main.cpp: CLI dispatch, argument validation and exit codes.
- src/common.h / common.cpp: move-only handles, UTF-8 files, errors, QPC.
- src/processes.cpp: minimal-rights process/token inspection, parents,
  modules and threads.
- src/runner.cpp: complete INI validation, jobs/IOCP, explicit inheritance,
  reader threads, limits, reasons, accounting and E1.
- kit/t3_child.c, check_t3.py, experiment_rules.py, test_jobs.ini: unchanged
  supplied course files.
- kit/validate_e1.py: labelled adapter for the supplied E1 rule module.
- kit/validate_extra.py: additional lifecycle/configuration tests.
- AI_USAGE.md and EXPERIMENTS.md: editable templates with AI-run evidence.

## Design
Normal builds use C:\Faculty\CSSO\T3. run uses CREATE_NO_WINDOW;
conhost may count toward job process limits. E1 uses DETACHED_PROCESS and
PROCESS_MEMORY, while ordinary jobs use JOB_MEMORY.

The inherited handle array stays alive until CreateProcessW returns. Only
stdout/stderr write ends are inherited; stdin is NULL. Parent copies of both
write ends close immediately after launch. Two Win32 reader threads drain
concurrently, preserving only the first 1 MiB but counting every byte.

The runner waits for ACTIVE_PROCESS_ZERO, not just the main process. An
absolute wall deadline is enforced even under continuous IOCP traffic.
Memory notifications alone do not kill the child; reasons use R6 precedence.

Move-only handles provide scope cleanup. Errors after assignment terminate
the whole job; the sole TerminateProcess call handles an unassigned suspended
process. Reader cleanup uses CancelSynchronousIo and waits, never
TerminateThread. The job normally closes last.

## Actual validation — 31 October 2026
- MSVC /W4 /O2 compilation: passed with no warnings.
- Copied Visual Studio solution, Release x64: passed.
- Supplied check_t3.py: **42/42 passed**.
- Supplied E1 rules through the labelled adapter: **11/11 passed**.
- Additional checks: invalid later INI section launches nothing; invalid
  arguments/ranges; Unicode working directory; protected details returns 1;
  killing the supervisor terminates descendants: passed.
- E1 counts: 1491 (64/0), 2500 (64/1), 125 (1024/0), 2500 (1024/1).
  All children exited 0, all ActiveProcesses=0, all handle deltas=+3.

See results/check.txt, check_e1_rules.txt, check_extra.txt and raw TSV files.
Tests used the same source with T3_VALIDATION defined and T3_VALIDATION_ROOT
pointing inside the workspace. Only output/child paths and necessary INI
quoting changed; limits, scenarios, child source and checker assertions did
not. Ordinary Visual Studio/CMake builds do not enable that override.
The normal fixed-path executable was built but not executed: its output
directory lies outside this session's writable workspace.

The shared check_experiments.py, experiment_common.py, official
EXPERIMENTS_Template.md and Experiments_Guide.pdf were not supplied in the
T3 material. The full official experiment/Markdown checker has NOT run.
results/check_experiments.txt records this limitation, not a fabricated pass.
Run it from the course Kit when those files are available:

    python check_experiments.py T3 --md PATH_TO_PROJECT\EXPERIMENTS.md

## Before submission
Fill in your personal AI_USAGE and EXPERIMENTS sections honestly. AI predictions
and measurements are labelled; do not present them as your own advance
prediction. Leave the assessor's live variation untried.

Regenerate results with the normal Release build at the required path, and
run the official common experiment checker. The included logs are real
workspace validation, not a claim that this final fixed-path step occurred.
Flood captures and executable/build files are excluded from the archive.

Use T3/ of your faculty Gitea repository, make genuine commits as you work,
then tag and push the final T3 version yourself. No repository was changed
or submission made here. Rename the archive for your group/name.
Deadline: 3 November 2026, 23:59 Romanian time.
