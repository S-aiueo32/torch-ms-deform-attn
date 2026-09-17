# HF rebuilt with the candidate toolchain

On 2026-09-17, directly rebuilt the official HF CUDA source and the current
candidate with the same compiler and measured both on one Runpod L4. This
answers whether the remaining encoder forward gap comes from the published
HF build and quantifies the effect of disabling device metadata checks.

**For encoder FP32 forward, the rebuilt HF and published HF have effectively
the same latency. Removing the candidate's metadata checks helps, but leaves
a smaller GPU-side difference. The focused follow-up identifies the forward
launch-bound annotation as a cause of that remaining unchecked-path gap.**
The main comparison and a focused follow-up
are stored with the source and binary hashes in the evidence archive.

## Experiment scope

The candidate in this report predates the public
[metadata switch](../direct-20260917-toggle/README.md).
`candidate_unchecked` changes only the copied `valid_spatial_level` body to
`return true`, removing content checks before feature access. Host checks and
index-range selection remain. Fixed benchmark metadata is prevalidated, and
outputs and all three gradients must pass an independent FP64 oracle.

PyTorch's `CUDA_KERNEL_ASSERT` remains active under `NDEBUG`; removing only the
assertion would also leave the helper's comparisons and early returns.

## Matched rebuild

All three diagnostic extensions use NVCC 12.6.85, GCC 11.4, PyTorch
2.10.0+cu126, C++17, sm_89 and `-O3`, with no fast-math or lineinfo flag.
HF and candidate are built with the same `cpp_extension.load` route and
minimal bindings. Their CUDA files are unmodified; only the unchecked copy
has the documented helper substitution. The normal candidate wheel is also
measured to check the alternate build route. Every native backend uses the
same diagnostic autograd wrapper and FP16-to-FP32 compute policy.

The published HF artifact remains pinned at
`abfd4042216fa4f84c9c5c4e3e844a3143c70ad5`, binary SHA256
`1a5053e022f3e5840ca9f40d361b5356ef0beeefdd8686029378dd4a7284b959`.
The official source's CUDA and wrapper files have no changes after their
initial addition, and the wrapper matches the packaged Python code byte for
byte. The older artifact does not identify its original build commit, so
its later metadata-signing commit is not treated as that build commit.
[Source provenance](hf-source-provenance.json) and
[reproduction procedure](REPRODUCE.md).

## Main comparison

Six backends × two production shapes × two dtypes: all 24 FP64 output and
all-gradient gates pass. There are 336 measurements across forward/training
and seven shuffled repetitions, with 0.6-second wall timing windows. Input
creation, qualification, eager execution and graph replay share one dedicated
nondefault stream. No profiler or concurrent builds run during timing.

Encoder FP32 forward, microseconds:

| Backend | Wall median | CUDA Graph median | Graph repetition range |
| --- | ---: | ---: | ---: |
| Candidate wheel public API | 602.64 | 599.48 | 599.07–599.69 |
| Candidate rebuilt, checked | 602.77 | 599.42 | 597.54–599.94 |
| Candidate rebuilt, unchecked | 582.04 | 577.21 | 577.08–583.24 |
| HF rebuilt | 571.46 | 566.87 | 566.83–567.78 |
| HF published | 571.99 | 566.98 | 566.86–567.90 |

The unchecked change reduces the graph median by 22.21 µs (3.7%). Matched
repetition improvements are 16.12–22.82 µs, all in the same direction. Its
remaining gap against rebuilt HF is 10.34 µs (1.8%), with matched differences
9.30–16.37 µs. Published/rebuilt HF graph medians differ by 0.11 µs; paired
differences range from −1.00 to +0.80 µs. The artifact build route is therefore
not the main explanation for this encoder forward gap.

The 22.21 µs reduction is the total effect of removing the check, including
changed loads, branches, register allocation and generated code. It is not
the isolated cost of comparison instructions. Graph intervals include output
initialization, all captured kernels and replay gaps.

FP16 must be considered separately because casts and captured allocation
state are part of the workload:

| Encoder FP16 forward | Wall µs | Graph µs |
| --- | ---: | ---: |
| Candidate rebuilt, checked | 727.90 | 719.26 |
| Candidate rebuilt, unchecked | 712.83 | 715.21 |
| HF rebuilt | 696.65 | 689.59 |
| HF published | 697.08 | 688.69 |

Do not apply the FP32 22/10 µs breakdown to FP16 or other shapes. Decoder
training wall/graph results have wider variation; the full distributions
are retained rather than converted into a universal speedup claim.

[Main summary](summary.json). Absolute times differ from earlier Pod runs;
all causal comparisons above use the same process and GPU in this run.

## Generated code

The exact sm_89 FP32 forward binaries have these static instruction/register
counts: checked candidate 304/42, unchecked candidate 232/41, and HF 240/39.
Published/rebuilt HF counts agree, as do wheel/rebuilt candidate counts.
The unchecked candidate has fewer instructions than HF but runs slower;
static totals alone do not explain latency. Hardware counters were unavailable.
[Binary analysis](binary-summary.json).

## Follow-up: the remaining unchecked-path difference

The three remaining source differences were changed independently, keeping
the reference builds intact. The comparison uses a fresh process, seven
backends, encoder forward, both dtypes and seven shuffled repetitions. All
14 output/all-gradient gates pass; 98 measurements complete without errors.

CUDA Graph medians in microseconds:

| Backend | FP32 | FP16 |
| --- | ---: | ---: |
| Candidate rebuilt, checked | 599.14 | 727.07 |
| Candidate unchecked | 583.09 | 708.53 |
| Unchecked, HF loop arithmetic | 582.38 | 704.65 |
| Unchecked, forward launch bound removed | 561.87 | 689.40 |
| HF rebuilt | 567.26 | 701.26 |
| HF rebuilt, forward zeros → empty | 559.90 | 678.66 |
| HF published | 567.04 | 696.42 |

Removing only the unchecked candidate's forward `__launch_bounds__(1024)`
improves FP32 by 21.22 µs (3.64%), and FP16 by 19.13 µs (2.70%). All seven
matched repetitions improve for each dtype. The actual launch remains 1024
threads per block; interpolation and floating-point expressions do not change.
This establishes a causal effect of the launch-bound annotation on the
unchecked forward's generated code/performance under these conditions.

The resulting FP32 kernel has 240 static instructions, 148 integer
instructions and 39 registers, matching the HF summaries. The unchecked
baseline has 232/151/41. Matching summaries are not proof of byte-identical
machine code, and neither the counts nor this experiment identify a hardware
stall or achieved-occupancy mechanism. In particular, fewer total static
instructions were not sufficient to make the unchecked baseline faster.

Changing only the grid-loop arithmetic produces the same opcode summary as
the unchecked baseline and no reliable FP32 improvement: paired differences
range from −12.23 to +9.41 µs. It is not supported as the explanation here.

HF's output zero-fill adds work. Removing it improves FP32 graph latency by
7.37 µs, in all seven pairs, without changing its forward-kernel instruction
summary. The no-bound candidate and HF-empty FP32 results are close: aggregate
medians differ by 1.97 µs, but the median paired difference is only 0.08 µs
and wins split three/four. This does not require assuming a cache benefit from
the extra zero-fill. No cache/power-counter attribution is made.

The result is not universal parity. FP16 graph still favors HF-empty by
10.73 µs, although their wall medians are almost equal (700.11/700.20 µs).
Also, these launch-bound tests start from the **unchecked** candidate. The
effect on the checked production kernel has not been measured, and removing
bounds indiscriminately could regress launchability for FP64 or int64 indexing.
The existing bounds were added to constrain register allocation on those
paths. Any production optimization must preserve those paths and be validated
separately; the diagnostic loop substitution is unsuitable for large indices.

[Follow-up summary](followup-summary.json),
[binary comparison](followup-binary.json),
[three complete forward profiles](followup-profiles.json).

## Saved evidence

[rebuild-evidence.tar.gz](rebuild-evidence.tar.gz) contains the official HF
source/provenance, exact candidate snapshot, all builders and measurement
scripts, raw reports and repetition samples, build commands, traces and binary
dumps. The embedded `results/evidence-manifest.json` lists all archived files.
After download, all 187 listed member hashes were verified with zero
mismatches. [Verification record](evidence-verification.json).

Archive SHA256:
`a984198d86a6b515ddba079ebe1580766c5f1177f7359acc02ed4eab205bef38`.
No public API or production source changed in this experiment.

Both rented Pods are deleted: [attempt 1](runpod-attempt1-cleanup.json) was
cleaned up after a transient Runpod API failure during startup;
[attempt 2](runpod-attempt2-cleanup.json) completed the experiments. Deletion of
each recorded Pod identifier was independently verified by the controller.
