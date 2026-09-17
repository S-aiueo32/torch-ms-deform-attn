#!/usr/bin/env python3
"""Inspect the exact measured CUDA binaries with cuobjdump (no network access).

python collect_binary.py RESULTS/report.json --variants-dir VARIANTS

Raw resource/SASS output is gzip-compressed. binary_summary.json/md contain
FP32 forward and C=32 backward resources and *static* instruction counts.
Static counts are not dynamic counts, runtime traffic, or timing attribution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

DEFAULT_BACKENDS = "canonical,hf,baseline,int32,no_metadata,int32_no_metadata"
INTEGER_OPS = {
    "IADD",
    "IADD3",
    "IMAD",
    "IMUL",
    "IDP",
    "IDP4A",
    "ISCADD",
    "ISETP",
    "ISET",
    "SHF",
    "SHL",
    "SHR",
    "LOP",
    "LOP3",
    "LEA",
    "BMSK",
    "BFI",
    "BFE",
    "FLO",
    "POPC",
    "BREV",
    "UIADD3",
    "UIMAD",
    "UISETP",
    "ULOP3",
    "USHF",
    "ULEA",
    "UFLO",
}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def find_binaries(report, report_path, variants_dir, requested):
    candidates = []
    canonical = report.get("canonical_binary")
    if canonical and "canonical" in requested:
        candidates.append({"backend": "canonical", **canonical})
    if "hf" in requested:
        hf_module = report.get("hf", {}).get("module", {})
        for item in hf_module.get("files", []):
            if item:
                path = Path(item.get("path", ""))
                is_elf = False
                if path.is_file():
                    with path.open("rb") as handle:
                        is_elf = handle.read(4) == b"\x7fELF"
                # Hub snapshot symlinks resolve to extensionless content blobs.
                if path.suffix == ".so" or is_elf:
                    candidates.append({"backend": "hf", **item})
        if not any(item["backend"] == "hf" for item in candidates):
            path = hf_module.get("file")
            if path and Path(path).suffix == ".so":
                candidates.append({"backend": "hf", "path": path})
    manifest = report.get("variants_manifest", {})
    if variants_dir and (variants_dir / "manifest.json").is_file():
        manifest = json.loads((variants_dir / "manifest.json").read_text())
    variants = manifest.get("variants", []) if isinstance(manifest, dict) else manifest
    if isinstance(variants, dict):
        variants = [
            {"name": name, **entry} for name, entry in variants.items() if isinstance(entry, dict)
        ]
    for variant in variants:
        name = variant["name"]
        if name not in requested:
            continue
        path = variant.get("path", variant.get("so_path"))
        if path:
            candidates.append({"backend": name, "path": path})
    resolved = []
    seen = set()
    for candidate in candidates:
        original = Path(candidate["path"])
        possibilities = [original, report_path.parent / original]
        if variants_dir and candidate["backend"] not in ("canonical", "hf"):
            root = variants_dir / candidate["backend"]
            possibilities.extend((variants_dir / original, root / "build" / original.name))
        path = next((path for path in possibilities if path.is_file()), original)
        key = (candidate["backend"], str(path))
        if key not in seen:
            seen.add(key)
            resolved.append({**candidate, "path": str(path)})
    return resolved


def demangle(symbols):
    executable = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if executable:
        result = subprocess.run(
            [executable, "-n"],
            input="\n".join(symbols) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        lines = result.stdout.splitlines()
        if not result.returncode and len(lines) == len(symbols):
            return dict(zip(symbols, lines))
    return dict(zip(symbols, symbols))


def relevant(symbol, name):
    fp32 = "<float" in name or bool(re.search(r"_kernel[^I]*If", symbol))
    if not fp32:
        return None
    if "ms_deformable_im2col_gpu_kernel" in name:
        match = re.search(r"im2col_gpu_kernel<float,\s*([^>]+)>", name)
        index_type = match.group(1) if match else None
        if index_type is None:
            match = re.search(r"im2col_gpu_kernelIf([ilx])E", symbol)
            index_type = (
                {"i": "int", "l": "long", "x": "long long"}.get(match.group(1)) if match else None
            )
        return {"kind": "forward", "scalar": "float", "index_type_template": index_type}
    if "ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce" in name:
        match = re.search(r"blocksize_aware_reduce_v\d<float,\s*(\d+)(?:u|ul)?>", name)
        if not match:
            match = re.search(r"blocksize_aware_reduce_v\dIfLj(\d+)E", symbol)
        if match and int(match.group(1)) == 32:
            return {"kind": "backward", "scalar": "float", "block_size": 32}
    return None


def parse_resources(text, default_arch=None):
    records = {}
    architecture = default_arch
    current = None
    for line in text.splitlines():
        arch_match = re.search(r"\barch\s*=\s*(sm_\d+[a-z]?)", line)
        if arch_match:
            architecture = arch_match.group(1)
            current = None
        function = re.match(r"\s*Function\s*:?[ \t]+(\S+?)\s*:?\s*$", line)
        if function:
            symbol = function.group(1).removesuffix(":")
            current = records.setdefault(
                (architecture, symbol), {"architecture": architecture, "symbol": symbol}
            )
            continue
        if current is not None:
            for field, value in re.findall(
                r"\b(REG|STACK|SHARED|LOCAL|TEXT|CONSTANT\[\d+\])\s*:\s*(\d+)", line
            ):
                current[field.lower()] = int(value)
    return records


def parse_sass(text, default_arch=None):
    records = {}
    architecture = default_arch
    current = None
    instruction_pattern = re.compile(
        r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9_.]*)\b(.*?);"
    )
    for line in text.splitlines():
        arch_match = re.search(r"\barch\s*=\s*(sm_\d+[a-z]?)", line)
        if arch_match:
            architecture = arch_match.group(1)
            current = None
        function = re.match(r"\s*Function\s*:\s*(\S+)\s*$", line)
        if function:
            symbol = function.group(1)
            current = records.setdefault(
                (architecture, symbol),
                {
                    "architecture": architecture,
                    "symbol": symbol,
                    "instructions": [],
                },
            )
            continue
        instruction = instruction_pattern.match(line)
        if instruction and current is not None:
            address, opcode, operands = instruction.groups()
            current["instructions"].append(
                {"address": address, "opcode": opcode, "operands": operands.strip()}
            )
    for record in records.values():
        instructions = record.pop("instructions")
        opcodes = Counter(item["opcode"] for item in instructions)
        base_opcodes = Counter()
        for opcode, count in opcodes.items():
            base_opcodes[opcode.split(".")[0]] += count
        record["static_instruction_count"] = len(instructions)
        record["static_opcode_counts"] = dict(sorted(opcodes.items()))
        record["static_base_opcode_counts"] = dict(sorted(base_opcodes.items()))
        record["static_integer_instruction_count"] = sum(
            count for opcode, count in base_opcodes.items() if opcode in INTEGER_OPS
        )
        record["static_iadd3_count"] = base_opcodes["IADD3"]
        record["static_imad_count"] = base_opcodes["IMAD"]
        record["static_imad_wide_count"] = sum(
            count
            for opcode, count in opcodes.items()
            if opcode == "IMAD.WIDE" or opcode.startswith("IMAD.WIDE.")
        )
        record["static_local_load_count"] = base_opcodes["LDL"]
        record["static_local_store_count"] = base_opcodes["STL"]
        record["static_branch_count"] = base_opcodes["BRA"]
        record["static_barrier_count"] = base_opcodes["BAR"]
    return records


def choose_option(help_text, long_option, short_option):
    if long_option in help_text:
        return long_option
    if short_option in help_text:
        return short_option
    raise RuntimeError(f"cuobjdump does not advertise {long_option} or {short_option}")


def dump(executable, option, architecture_args, binary, output):
    command = [executable, option, *architecture_args, str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        handle.write(result.stdout)
    stderr = output.with_suffix(".stderr.txt")
    stderr.write_text(result.stderr)
    return result.stdout, {
        "command": command,
        "returncode": result.returncode,
        "stdout_gzip": str(output),
        "stderr": str(stderr),
    }


def render(summary):
    lines = [
        "# Exact-binary CUDA inspection",
        "",
        "Static SASS instruction counts describe the compiled instruction listing, not executed instructions, dynamic spill traffic, or elapsed-time attribution. Loops, branch paths and predication change runtime counts.",
        "",
        "Register and local-memory sizes are compiled resource declarations. LDL/STL indicate local-memory instructions; local memory can have purposes besides spills.",
        "",
        f"Requested architecture: `{summary['requested_architecture']}`. Only that architecture is included in the comparison table.",
        "",
        "| Backend | Kernel | Architecture | Registers | Stack bytes | Local bytes | Static inst. | Integer | IADD3 | IMAD | IMAD.WIDE | LDL | STL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for binary in summary["binaries"]:
        for kernel in binary.get("kernels", []):
            if kernel.get("architecture") != summary["requested_architecture"]:
                continue
            label = (
                "fwd, template index=" + str(kernel.get("index_type_template") or "absent")
                if kernel["kind"] == "forward"
                else "bwd, C=32"
            )
            resources = kernel.get("resources", {})
            values = [
                binary["label"],
                label,
                kernel.get("architecture") or "?",
                *(str(resources.get(field, "?")) for field in ("reg", "stack", "local")),
                *(
                    str(kernel.get(field, "?"))
                    for field in (
                        "static_instruction_count",
                        "static_integer_instruction_count",
                        "static_iadd3_count",
                        "static_imad_count",
                        "static_imad_wide_count",
                        "static_local_load_count",
                        "static_local_store_count",
                    )
                ),
            ]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Binary provenance", ""]
    for binary in summary["binaries"]:
        observed = binary.get("fingerprint", {})
        lines += [
            f"- `{binary['label']}`: `{binary['path']}`; SHA256 `{observed.get('sha256', '?')}`; measured-report hash match: `{binary.get('matches_report_sha256')}`."
        ]
        for warning in binary.get("warnings", []):
            lines.append(f"  - {warning}")
    if summary.get("missing_backends"):
        lines += ["", "No binary recorded for: " + ", ".join(summary["missing_backends"]) + "."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--variants-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.report.parent / "binary_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(args.cuobjdump)
    if not executable:
        parser.error(f"cuobjdump executable not found: {args.cuobjdump}")
    help_result = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, check=False
    )
    help_text = help_result.stdout + help_result.stderr
    (output_dir / "cuobjdump-help.txt").write_text(help_text)
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    (output_dir / "cuobjdump-version.txt").write_text(version.stdout + version.stderr)
    resource_option = choose_option(help_text, "--dump-resource-usage", "-res-usage")
    sass_option = choose_option(help_text, "--dump-sass", "-sass")
    architecture_args = []
    if "--gpu-architecture" in help_text:
        architecture_args = ["--gpu-architecture", args.arch]
    elif re.search(r"(?<!\S)-arch(?:\s|,|$)", help_text):
        architecture_args = ["-arch", args.arch]
    report = json.loads(args.report.read_text())
    requested = args.backends.split(",")
    candidates = find_binaries(report, args.report, args.variants_dir, requested)
    summary = {
        "report": str(args.report.resolve()),
        "requested_architecture": args.arch,
        "architecture_cli_filter_supported": bool(architecture_args),
        "cuobjdump": executable,
        "version": version.stdout.strip(),
        "notes": [
            "Static instruction counts must not be interpreted as dynamic counts or runtime attribution.",
            "IMAD.WIDE is included in IMAD; integer total includes explicitly listed integer/bit-operation families.",
            "LDL/STL are local-memory instructions, not proof that all local traffic represents spills.",
            "A missing native sm_89 cubin can mean the driver selects another architecture or JIT-compiles PTX; this script cannot identify JIT output.",
            "Canonical/HF hashes are checked against probe fingerprints; variant hashes were not recorded at measurement time and are reported as observed now.",
        ],
        "integer_opcode_families": sorted(INTEGER_OPS),
        "missing_backends": [
            name for name in requested if not any(item["backend"] == name for item in candidates)
        ],
        "binaries": [],
    }
    counts = Counter(item["backend"] for item in candidates)
    ordinal = Counter()
    for candidate in candidates:
        backend = candidate["backend"]
        ordinal[backend] += 1
        label = backend if counts[backend] == 1 else backend + "_" + str(ordinal[backend])
        path = Path(candidate["path"])
        entry = {"backend": backend, "label": label, "path": str(path), "warnings": []}
        summary["binaries"].append(entry)
        if not path.is_file():
            entry["warnings"].append("Recorded binary path does not exist.")
            continue
        print(f"Inspecting {label}: {path}", flush=True)
        entry["fingerprint"] = fingerprint(path)
        entry["expected_report_sha256"] = candidate.get("sha256")
        entry["matches_report_sha256"] = (
            candidate.get("sha256") == entry["fingerprint"]["sha256"]
            if candidate.get("sha256")
            else None
        )
        if entry["matches_report_sha256"] is False:
            entry["warnings"].append(
                "BINARY CHANGED since measurement: do not attribute this disassembly to the recorded timings."
            )
        resources_text, resource_dump = dump(
            executable,
            resource_option,
            architecture_args,
            path,
            output_dir / (label + ".resources.txt.gz"),
        )
        sass_text, sass_dump = dump(
            executable, sass_option, architecture_args, path, output_dir / (label + ".sass.txt.gz")
        )
        entry["dumps"] = {"resources": resource_dump, "sass": sass_dump}
        for kind, dump_result in entry["dumps"].items():
            if dump_result["returncode"]:
                entry["warnings"].append(
                    f"{kind} dump failed with exit code {dump_result['returncode']}; see saved stderr."
                )
        default_arch = args.arch if architecture_args else None
        resources = parse_resources(resources_text, default_arch)
        sass = parse_sass(sass_text, default_arch)
        symbols = sorted({key[1] for key in resources} | {key[1] for key in sass})
        demangled = demangle(symbols)
        kernels = []
        for key in sorted(set(resources) | set(sass), key=lambda key: (key[0] or "", key[1])):
            architecture, symbol = key
            classification = relevant(symbol, demangled[symbol])
            if not classification:
                continue
            kernels.append(
                {
                    "architecture": architecture,
                    "symbol": symbol,
                    "demangled": demangled[symbol],
                    **classification,
                    **sass.get(key, {}),
                    "resources": resources.get(key, {}),
                }
            )
        entry["kernels"] = kernels
        if not any(
            item["architecture"] == args.arch and item.get("static_instruction_count")
            for item in kernels
        ):
            entry["warnings"].append(
                f"No relevant native {args.arch} SASS found; loaded device machine code is not established by this inspection."
            )
        (output_dir / "binary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "binary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render(summary)
    (output_dir / "binary_summary.md").write_text(text)
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
