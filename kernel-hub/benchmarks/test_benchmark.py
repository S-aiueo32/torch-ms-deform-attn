"""CPU checks of benchmark qualification, independent of CUDA availability."""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark
import torch
from benchmark import CASES, inputs, matched_precision, validate

from torch_ms_deform_attn import _C, ms_deform_attn, ms_deform_attn_core_pytorch


class QualificationTest(unittest.TestCase):
    def test_full_driver_on_cpu_with_external_backends_substituted(self):
        """Exercise orchestration/serialization, not competitor GPU correctness."""

        def load_local(path):
            self.assertIsInstance(path, Path)
            return types.SimpleNamespace(ms_deform_attn=ms_deform_attn)

        hf = types.SimpleNamespace(
            ms_deform_attn_forward=_C.ms_deform_attn_forward,
            ms_deform_attn_backward=_C.ms_deform_attn_backward,
        )

        def inplace_backward(v, s, t, loc, w, g, gv, gl, gw, step):
            for dest, src in zip(
                (gv, gl, gw), _C.ms_deform_attn_backward(v, s, t, loc, w, g, step)
            ):
                dest.copy_(src)

        def triton(v, shapes, loc, weights, padding, align):
            return ms_deform_attn_core_pytorch(v, shapes, loc, weights).reshape(
                v.shape[0], loc.shape[1], v.shape[2], v.shape[3]
            )

        def measure(run, args):
            output = run()
            values = (output,) if isinstance(output, torch.Tensor) else output
            self.assertTrue(all(torch.isfinite(value).all() for value in values))
            return {"wall_ms": 0.0}  # Synthetic timing never leaves this temporary test.

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources.json").write_text("{}")
            argv = [
                "benchmark",
                "--sources",
                tmp,
                "--output",
                str(root / "result.json"),
                "--repeats",
                "2",
                "--warmup",
                "1",
                "--native-control",
            ]
            frontend = types.ModuleType("msda_triton.frontend")
            frontend.triton_multiscale_deformable_attention = triton

            # Restore only this entry: Torch lazily imports registered operators,
            # which must not be removed from sys.modules when the test exits.
            @contextlib.contextmanager
            def substitute_frontend():
                name = frontend.__name__
                previous = sys.modules.get(name)
                sys.modules[name] = frontend
                try:
                    yield
                finally:
                    if previous is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = previous

            with (
                patch.dict(CASES, {"tiny": (1, 3, 2, 4, ((4, 4), (2, 2)))}, clear=True),
                patch.object(sys, "argv", argv),
                patch.dict(
                    os.environ, {"MSDA_HF_BASELINE_REVISION": "0" * 40, "MSDA_KERNEL_DIR": tmp}
                ),
                substitute_frontend(),
                patch.object(benchmark, "get_kernel", return_value=hf),
                patch.object(benchmark, "get_local_kernel", side_effect=load_local),
                patch.object(
                    benchmark,
                    "load_mmcv",
                    return_value=types.SimpleNamespace(
                        forward=_C.ms_deform_attn_forward, backward=inplace_backward
                    ),
                ),
                patch.object(
                    benchmark, "inputs", side_effect=lambda c, d: inputs(c, d, device="cpu")
                ),
                patch.object(benchmark, "environment", return_value={"device": "cpu-test"}),
                patch.object(benchmark, "measure", side_effect=measure),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(benchmark.main(), 0)
                argv[argv.index("--output") + 1] = str(root / "focused.json")
                argv.extend(
                    ["--backends", "torch-ms-deform-attn", "hf-native", "upstream-native-control"]
                )
                with patch.object(benchmark, "load_mmcv", side_effect=RuntimeError("unavailable")):
                    self.assertEqual(benchmark.main(), 0)
                focused = json.loads((root / "focused.json").read_text())
                self.assertEqual(
                    {r["backend"] for r in focused["results"]},
                    {"torch-ms-deform-attn", "hf-native", "upstream-native-control"},
                )
            report = json.loads((root / "result.json").read_text())
            qualified = [r for r in report["results"] if r["policy"] == "fp32-compute"]
            self.assertEqual(len(qualified), 7 * 3 * 3 * 2)
            self.assertEqual({r["repeat"] for r in qualified}, {0, 1})
            self.assertTrue(all(r["status"] == "passed" for r in qualified))
            self.assertTrue(
                all("wall_ms" not in r for r in report["results"] if r["status"] != "passed")
            )

    def test_precision_and_gradient_validation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                patch.dict(CASES, {"tiny": (1, 3, 2, 4, ((4, 4), (2, 2)))}),
            ):
                torch.manual_seed(29)
                data, grad, sizes, scale = inputs("tiny", dtype, device="cpu")

                def reference(value, shapes, starts, loc, weights):
                    return ms_deform_attn_core_pytorch(value, sizes, loc, weights)

                oracle_inputs = tuple(
                    tensor.detach().double().requires_grad_()
                    for tensor in (data[0], data[3], data[4])
                )
                out = reference(oracle_inputs[0], data[1], data[2], *oracle_inputs[1:])
                truth = (out, *torch.autograd.grad(out, oracle_inputs, grad.double()))
                self.assertTrue(
                    validate(matched_precision(reference), data, grad, truth, scale)["passed"]
                )

                # A forward-correct implementation with a broken coordinate
                # gradient must not qualify for timing.
                def broken(value, shapes, starts, loc, weights):
                    frozen = loc.detach() + loc * 0
                    return matched_precision(reference)(value, shapes, starts, frozen, weights)

                result = validate(broken, data, grad, truth, scale)
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"]["output"]["close"])
                self.assertFalse(result["errors"]["grad_locations"]["close"])


if __name__ == "__main__":
    unittest.main()
