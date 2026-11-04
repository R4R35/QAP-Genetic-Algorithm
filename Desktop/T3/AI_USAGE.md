# AI_USAGE — T3
Student: [name] | Group: [group] | Date: [submission date]

## Tools and purposes
ChatGPT / Codex read the assignment and public tests, generated the C++
implementation, integrated a copy of the existing Visual Studio solution,
compiled and executed Windows tests, and prepared these documentation
templates. AI assistance included substantial code generation. The course
child and checker files were reused unchanged.

## Checks performed by the AI assistant
MSVC /W4 /O2 compilation and Visual Studio Release x64 build passed.
The supplied check_t3.py passed 42/42 checks. All four E1 combinations ran;
the supplied E1 rule module passed 11/11 checks through a labelled adapter.
Additional tests covered invalid configuration before execution, Unicode
working directory, protected process inspection and KILL_ON_JOB_CLOSE.
Actual logs and measurements are in results/.

Execution used the same source with a compile-time output-path override
inside the writable workspace. The normal fixed-path executable was built
but needs its final run at C:\Faculty\CSSO\T3. The shared official experiment/
Markdown checker was unavailable and was not reported as passed.

## Corrections during validation
- Kept the explicit inherited-handle array alive through CreateProcessW.
- Queried TokenElevation with the exact TOKEN_ELEVATION buffer size.
- Configured supplemental Python output as UTF-8.
No checker assertion or expected result was weakened.

## My review and understanding — complete honestly
[Which functions did you inspect and understand? Which tests did you rerun?
What changes did you personally make? Do not claim checks you have not done.]
[Briefly explain closing the parent's write handles, launching suspended,
and why waiting only for the main process is insufficient.]

## Responsibility
[Confirm after reviewing: I understand the implementation and can explain
its resource ownership, wait loop, error handling and experiments.]
