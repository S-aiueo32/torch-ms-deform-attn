# C++ and CUDA checks

[Contributing](../CONTRIBUTING.md) · [Development](development.md)

The Metal backend also contains Objective-C++ (`csrc/mps/ms_deform_attn_mps.mm`)
and embedded shader source (`csrc/mps/kernels.h`). Include them and the MPS header
in formatting checks. For cpplint, add `--extensions=h,cpp,mm` to include the bridge.
The shader source inside the raw string is checked by actual Metal compilation.
Use `--header-filter` to restrict clang-tidy diagnostics to this repository's
`csrc/` headers; PyTorch/system headers otherwise produce unrelated warnings.

Use these checks when changing `csrc/`. First complete the
[development setup](development.md#set-up-with-uv) and rebuild the extension.
The clang-tidy recipe below is specific to macOS; Python checks and runtime tests
are described in the development guide.

## Format C++ and CUDA

With `clang-format` available through `xcrun` on macOS, run the command below.
On other platforms, invoke `clang-format` directly with the same file list:

```bash
xcrun clang-format -i csrc/cuda/index_utils.h \
  csrc/cuda/ms_deform_attn_cuda.cu \
  csrc/cuda/ms_deform_attn_cuda.h \
  csrc/cuda/ms_deform_im2col_cuda.cuh \
  csrc/ms_deform_attn_cpu.cpp \
  csrc/ms_deform_attn_cpu.h \
  csrc/vision.cpp
```

## Run cpplint

Run cpplint with the repository's `runtime/references` filter:

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

## Run clang-tidy on macOS

`clang-tidy` needs `compile_commands.json`. This recipe uses macOS and Xcode
Command Line Tools to capture CPU compiler commands before checking the sources.

### 1. Create a compiler wrapper

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

### 2. Capture the build commands

```bash
: > /tmp/compile_commands_capture.jsonl
FORCE_CPU=1 CC_REAL=$(which c++) \
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

### 3. Run clang-tidy

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

CPU-only builds may produce only `.cpp` entries in `compile_commands.json`.
Review generated commands before running clang-tidy, and narrow its checks and
header filter when system headers produce unrelated warnings.
