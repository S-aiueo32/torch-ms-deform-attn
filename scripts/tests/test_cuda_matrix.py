"""Tests for the shared CUDA build and runtime configuration."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_matrix import CUDA_CONFIGS, TORCH_CONFIGS, build_matrix  # noqa: E402


class CUDAMatrixTest(unittest.TestCase):
    def test_runtime_mapping_matches_full_build_matrix(self):
        expected = {config["torch"]: (config["image"], config["cuda"]) for config in CUDA_CONFIGS}
        self.assertEqual(TORCH_CONFIGS, expected)

    def test_default_matrix_covers_minimum_and_latest(self):
        matrix = build_matrix(full=False)
        self.assertEqual(matrix, [CUDA_CONFIGS[0], CUDA_CONFIGS[-1]])
        self.assertIsNot(matrix[0], CUDA_CONFIGS[0])

    def test_cli_emits_compact_json(self):
        script = Path(__file__).resolve().parents[1] / "cuda_matrix.py"
        output = subprocess.check_output([sys.executable, script, "--full"], text=True)
        self.assertEqual(json.loads(output), list(CUDA_CONFIGS))
        self.assertEqual(output.count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
