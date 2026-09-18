"""Shared PyTorch, CUDA toolkit, and container matrix for CUDA CI."""

import argparse
import json

CUDA_CONFIGS = (
    {
        "torch": "2.4.0",
        "cuda": "12.4",
        "image": (
            "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel@sha256:"
            "e96c6896ecfbb50d89c87bf94110206ef444f27268c5f72201eb29fba9c90331"
        ),
    },
    {
        "torch": "2.5.1",
        "cuda": "12.4",
        "image": (
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:"
            "14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
        ),
    },
    {
        "torch": "2.7.1",
        "cuda": "12.6",
        "image": "pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel",
    },
    {
        "torch": "2.8.0",
        "cuda": "12.6",
        "image": (
            "pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel@sha256:"
            "adb0f2d3769e0796a5d86740b0f900a72835b6fa9577b77d6efed460f1322fcb"
        ),
    },
    {
        "torch": "2.14.0",
        "cuda": "12.6",
        "image": (
            "pytorch/pytorch:2.14.0-cuda12.6-cudnn9-devel@sha256:"
            "0e968d2570373aeb781603c21925a030053b432945967da14f6f8423e1f0cc79"
        ),
    },
)

TORCH_CONFIGS = {config["torch"]: (config["image"], config["cuda"]) for config in CUDA_CONFIGS}


def build_matrix(full: bool) -> list[dict[str, str]]:
    """Return independent dictionaries for the requested build matrix."""
    selected = CUDA_CONFIGS if full else (CUDA_CONFIGS[0], CUDA_CONFIGS[-1])
    return [dict(config) for config in selected]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="emit every supported pair")
    args = parser.parse_args(argv)
    print(json.dumps(build_matrix(args.full), separators=(",", ":")))


if __name__ == "__main__":
    main()
