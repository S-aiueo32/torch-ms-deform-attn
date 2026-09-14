# Contributing to torch-ms-deform-attn

This repository accepts changes to both Python and C++/CUDA code. Before opening a PR, please run the checks below so issues are fixed early.

## 1. Development setup

- Python 3.10+ (`.python-version` is 3.11 in CI)
- PyTorch 2.5.1 (`uv sync` keeps it consistent)
- A C++17 compiler (Xcode Command Line Tools + `xcrun`)
- Optional: CUDA, when validating CUDA build/runtime paths

```bash
uv sync
```

## 2. Build

Build the extension in CPU/CUDA mode for your environment (CUDA is included automatically if it is available):

```bash
.venv/bin/python setup.py build_ext --inplace --force
```

- Use `FORCE_CPU=1` or `FORCE_CUDA=1` to force a specific build target.
- When generating `compile_commands.json` (clang-tidy flow below), use the same environment settings as your normal build.

## 3. Python lint / format / type checks

These are the same as in `README.md` / `docs/development.md`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Auto-fixes:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## 4. C++ / CUDA formatting and static checks

### 4-1. clang-format

On macOS, `clang-format` is usually available via `xcrun`. Reformat the C++/CUDA sources listed below as needed:

```bash
xcrun clang-format -i csrc/cuda/index_utils.h \
  csrc/cuda/ms_deform_attn_cuda.cu \
  csrc/cuda/ms_deform_attn_cuda.h \
  csrc/cuda/ms_deform_im2col_cuda.cuh \
  csrc/ms_deform_attn_cpu.cpp \
  csrc/ms_deform_attn_cpu.h \
  csrc/vision.cpp
```

### 4-2. cpplint

Run with `runtime/references` filtered out, which is our current noise-reduction setup:

```bash
uvx --from cpplint cpplint --filter=-runtime/references --quiet \
  csrc/cuda/index_utils.h \
  csrc/cuda/ms_deform_attn_cuda.cu \
  csrc/cuda/ms_deform_attn_cuda.h \
  csrc/cuda/ms_deform_im2col_cuda.cuh \
  csrc/ms_deform_attn_cpu.cpp \
  csrc/ms_deform_attn_cpu.h \
  csrc/vision.cpp
```

### 4-3. clang-tidy

`clang-tidy` needs `compile_commands.json`. Generate it first in a CPU environment:

1) Create a compile wrapper once:

```bash
cat >/tmp/ccxx_capture.py <<'PY'
#!/usr/bin/env python3
import json, os, subprocess, shlex, sys
from pathlib import Path

real_cc = os.environ.get("CC_REAL", os.environ.get("CXX_REAL", "c++"))
log_path = Path(os.environ.get("CC_LOG", "/tmp/compile_commands_capture.jsonl"))
repo = Path(os.environ.get("REPO_ROOT", os.getcwd())).resolve()

args = list(sys.argv[1:])
if "-c" in args:
    source = None
    for arg in reversed(args):
        if arg.startswith("-"):
            continue
        if arg.lower().endswith((".cpp", ".cu", ".cxx", ".cc")):
            source = arg
            break
    if source:
        if "-I." not in args and "-I" + str(repo) not in args:
            args.extend(["-I", str(repo)])
        cmd = [real_cc] + args
        with log_path.open("a") as f:
            json.dump({
                "directory": str(Path.cwd()),
                "command": " ".join(shlex.quote(x) for x in cmd),
                "file": source,
            }, f)
            f.write("\n")
        ret = subprocess.run(cmd).returncode
    else:
        ret = subprocess.run([real_cc] + args).returncode
else:
    ret = subprocess.run([real_cc] + args).returncode
raise SystemExit(ret)
PY
chmod +x /tmp/ccxx_capture.py
```

2) Build through the wrapper and create `compile_commands.json`:

```bash
CC_REAL=$(which c++) \
CC=/tmp/ccxx_capture.py \
CXX=/tmp/ccxx_capture.py \
REPO_ROOT=$(pwd) \
.venv/bin/python setup.py build_ext --inplace --force

.venv/bin/python - <<'PY'
import json, pathlib
src = pathlib.Path('/tmp/compile_commands_capture.jsonl')
out = pathlib.Path('compile_commands.json')
cmds = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
commands = [
    c for c in cmds
    if c["file"].endswith(".cpp") and (c["file"].startswith("csrc/") or "/csrc/" in c["file"])
]
out.write_text(json.dumps(commands, indent=2))
PY
```

3) Run clang-tidy (narrow checks as needed):

```bash
uvx --from clang-tidy clang-tidy -p . \
  --checks='-*,bugprone-easily-swappable-parameters,readability-avoid-const-params-in-decls' \
  --header-filter="^$(pwd)/csrc/.*" \
  --extra-arg=-isysroot \
  --extra-arg=$(xcrun --show-sdk-path) \
  --extra-arg=-stdlib=libc++ \
  --extra-arg=-I$(xcrun --show-sdk-path)/usr/include/c++/v1 \
  csrc/ms_deform_attn_cpu.cpp csrc/vision.cpp
```

`-checks` should be adjusted for the repository policy so we keep only actionable warnings.

## 5. PR pre-checklist

- Run Python checks: `ruff`, `ty`
- Run C++/CUDA checks: `clang-format`, `cpplint`, `clang-tidy`
- Ensure `setup.py build_ext --inplace --force` succeeds
- Run relevant tests (for example, `unittest`)
- Review diffs if `compile_commands.json` changed

## 6. Notes

- `clang-tidy` can emit many environment-dependent warnings, especially from system headers; narrowing with `-checks` and `-header-filter` is recommended.
- On macOS, Xcode / Command Line Tools differences often require `-isysroot` and `-stdlib=libc++`.
- CPU-only environments may produce only `.cpp` entries under `csrc/` in `compile_commands.json`.
