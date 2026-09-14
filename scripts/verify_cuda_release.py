"""Check downloaded GPU release evidence against the release's full source SHA."""

import argparse
import json
import re
from pathlib import Path


def verify(directory, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Use the full release source SHA")
    report = json.loads((directory / "cuda-tests.json").read_text())
    completion = json.loads((directory / "cuda-completion.json").read_text())
    if report["source_sha"] != sha or completion["source_sha"] != sha:
        raise ValueError("GPU evidence does not match release SHA")
    if (
        not report["success"]
        or report["failures"]
        or report["errors"]
        or report["unexpected_skips"]
    ):
        raise ValueError("GPU suite failed or skipped unexpected tests")
    if (
        not report["gpu"]
        or not report["cuda_passed"]
        or report["tests_run"] <= len(report["skipped"])
    ):
        raise ValueError("No executed GPU suite")
    if not completion["success"]:
        raise ValueError("GPU validation did not complete")
    if not (directory / "cuda-checks.log").is_file():
        raise ValueError("Missing full validation log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    verify(args.evidence, args.sha)
    print(f"GPU release evidence verified for {args.sha}")


if __name__ == "__main__":
    main()
