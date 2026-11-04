# EXPERIMENTS — T3

Student: [your name]  
Group: [your group]  
Environment: Windows x64; Visual Studio 2022 / MSVC 19.42; Release /O2.  
Status: template with separately identified AI validation evidence.

## E1 — How many threads fit in a memory-limited job

### Question
How do stack size and STACK_SIZE_PARAM_IS_A_RESERVATION affect the number
of threads created under a 128 MiB per-process committed-memory limit?

### Student prediction — complete before your own first run
[Write your prediction and reasoning in your own words. The AI validation
below is not a prediction made by you. Do not backdate a prediction.]

| stack_kb | reservation | Your predicted count/range | Reason |
|---|---|---|---|
| 64 | 0 | [fill in] | [fill in] |
| 64 | 1 | [fill in] | [fill in] |
| 1024 | 0 | [fill in] | [fill in] |
| 1024 | 1 | [fill in] | [fill in] |

### AI validation prediction — recorded before AI-run E1 measurements
For reservation=0, CreateThread requests stack commit: 64 KiB should allow
roughly 1,000–2,000 threads and 1,024 KiB roughly 100–125 threads, below the
2,500 cap. The executable, heaps, thread bookkeeping, and other private
allocations also consume the limit, so simple division is only an upper bound.
For reservation=1, stack_kb determines address-space reservation, not an equally
large immediate commitment. Both settings should allow at least as many
threads as their non-reservation counterparts and may reach 2,500. Failure
8 or 1455 is possible before the cap because of committed-memory or other
system resource limits. These are predictions, not measured values.

### Method
Use the unchanged supplied t3_child.c, compiled for x64. Each run is launched
suspended, assigned to a separate job, and resumed with DETACHED_PROCESS.
Limits: PROCESS_MEMORY=128 MiB, ACTIVE_PROCESS=2, KILL_ON_JOB_CLOSE;
maximum 2,500 threads. Capture THREADS/CODE/PRIVATE_BEFORE/PRIVATE_AFTER,
wait for ACTIVE_PROCESS_ZERO, collect job accounting, and compare the parent
handle count before construction and after complete cleanup.

Commands (normal assignment build):
```text
t3.exe experiment threads 128 64 0
t3.exe experiment threads 128 64 1
t3.exe experiment threads 128 1024 0
t3.exe experiment threads 128 1024 1
```

### Measurements
AI-run measurements, 31 October 2026, 19:46:55 UTC (21:46:55 Romanian time).
These are actual measurements with output redirected into the workspace.

| stack_kb | reservation | threads | error | private/thread KiB | peak KiB | handles before/after | remaining |
|---|---|---|---|---|---|---|---|
| 64 | 0 | 1491 | 1455 | 87.4 | 131072 | 96 / 99 | 0 |
| 64 | 1 | 2500 | 0 | 29.1 | 73420 | 96 / 99 | 0 |
| 1024 | 0 | 125 | 1455 | 1035.4 | 130164 | 96 / 99 | 0 |
| 1024 | 1 | 2500 | 0 | 30.9 | 78036 | 96 / 99 | 0 |

All child exits: 0x00000000. Handle differences: +3 in all four cases.
The supplied E1 rules passed 11/11 through the supplemental adapter.
The official common/Markdown checker remains unavailable.

### AI interpretation of its measurements
The no-reservation counts matched the recorded prediction ranges:
1491 versus 125, a ratio of about 11.9. Committing larger stacks strongly
reduces capacity, but overhead means the ratio is not the naive factor 16.
Private increments were 87.4 and 1035.4 KiB, not exactly 64 and 1024 KiB.

Both reservation cases hit the 2500 cap with code 0. Their increments were
about 29–31 KiB per thread. Reserved address space is not all immediately
committed memory. Hitting the cap gives a lower bound on capacity, not the
actual maximum thread count.

1455 is ERROR_COMMITMENT_LIMIT. Its paging-file-related message can mislead:
the per-process job commit budget can reject allocations even if the whole
machine still has memory. The interpretation here uses the imposed 128 MiB
limit, observed peaks, and higher counts in the reservation cases together.

The constant +3 handle difference is within the supplied tolerance and does
not scale with thread count. It is consistent with fixed API/runtime
initialization, but these measurements do not identify those handles' owners.
ActiveProcesses=0 and the supplied live-process check found no surviving
experiment children.

### Your explanation
- Compare each prediction with your observed count, quoting the measured values.
- Explain commit versus reservation and why the initial committed stack is not
  necessarily the requested reservation size.
- Explain what else consumes private bytes and why private_per_thread_kib
  differs from stack_kb.
- Give the observed error code and its Windows message. Explain why a general
  memory/resources message does not prove that the computer's physical RAM is full.
- State whether each run hit the 2,500 cap or a resource limit.
- Quote handles_after - handles_before and remaining_processes for each run.

### Cleanup
[Record the handle differences and confirm that no experiment child remains.
Use the actual experiments.tsv evidence, not assumed zeros.]

## Live variation — leave untried until the assessor selects it
Parameters: [assessor's selection]
Prediction and reason, before running: [fill in at the presentation]
Measurement: [fill in after the run]
Explanation of the difference: [fill in]
