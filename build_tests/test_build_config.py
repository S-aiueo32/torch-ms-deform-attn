"""Build policy tests; real compilation is covered separately by CI."""

import contextlib
import io
import os
import runpy
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from setuptools import Distribution
from torch.utils.cpp_extension import BuildExtension

ROOT = Path(__file__).resolve().parents[1]


def load(**env):
    with (
        patch.dict(os.environ, env, clear=True),
        patch("setuptools.setup"),
        patch("torch.utils.cpp_extension.CUDA_HOME", None),
    ):
        return runpy.run_path(str(ROOT / "setup.py"))


class BuildConfigTest(unittest.TestCase):
    def test_missing_ninja_falls_back(self):
        ns = load(FORCE_CPU="1")
        with patch("torch.utils.cpp_extension.is_ninja_available", return_value=False):
            with self.assertWarnsRegex(UserWarning, "could not find ninja"):
                command = ns["OptionalOpenMPBuildExtension"](Distribution(), use_ninja=True)
        self.assertFalse(command.use_ninja)

    def test_invalid_selection(self):
        for env, message in (
            (dict(FORCE_CPU="1", FORCE_CUDA="1"), "cannot both"),
            (dict(FORCE_CUDA="1"), "CUDA-enabled PyTorch"),
            (dict(USE_NINJA="bad"), "USE_NINJA"),
        ):
            with self.subTest(env=env), self.assertRaisesRegex(RuntimeError, message):
                load(**env)

    def test_architecture(self):
        ns = load(FORCE_CPU="1")
        for flags in ("-arch wrong", "-arch"):
            with (
                patch.dict(os.environ, {"ARCHFLAGS": flags}),
                self.assertRaisesRegex(RuntimeError, "ARCHFLAGS"),
            ):
                ns["macos_target_flags"]()
        self.assertEqual(
            ns["without_macos_arch_flags"](["c++", "-arch", "arm64", "-O3"]), ["c++", "-O3"]
        )

    def command(self):
        ns = load(FORCE_CPU="1")
        cls = ns["OptionalOpenMPBuildExtension"]
        cmd = cls(Distribution(), use_ninja=False)
        cmd.compiler = SimpleNamespace(compiler_type="unix", compiler_so=["c++"])
        cmd.extensions = [SimpleNamespace(extra_compile_args={"cxx": []}, extra_link_args=[])]
        return ns, cmd

    def test_backend_selection(self):
        for backend, setting, failure, expected in (
            ("openmp", None, None, "openmp"),
            ("openmp", None, RuntimeError("probe failed"), "serial"),
            ("openmp", None, subprocess.TimeoutExpired("c++", 30), "serial"),
            ("openmp", "0", None, "serial"),
            ("native", None, None, "native"),
            ("serial", None, None, "serial"),
        ):
            with self.subTest(backend=backend, setting=setting, failure=failure):
                ns, cmd = self.command()
                env = {} if setting is None else {"FORCE_OPENMP": setting}
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(
                        ns["aten_parallel_backend"].__globals__,
                        {"aten_parallel_backend": lambda: backend},
                    ),
                    patch.object(
                        cmd, "openmp_flags", side_effect=failure, return_value=(["-fopenmp"], [])
                    ),
                    patch.object(BuildExtension, "build_extensions") as build,
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    cmd.build_extensions()
                self.assertIn(f"backend: {expected}", output.getvalue())
                self.assertTrue(cmd.force)
                build.assert_called_once()

    def test_required_openmp_errors(self):
        for backend, setting, message in (
            ("openmp", "1", "could not enable"),
            ("native", "1", "requires an OpenMP"),
            ("openmp", "bad", "FORCE_OPENMP must"),
        ):
            ns, cmd = self.command()
            with (
                patch.dict(os.environ, {"FORCE_OPENMP": setting}, clear=True),
                patch.dict(
                    ns["aten_parallel_backend"].__globals__,
                    {"aten_parallel_backend": lambda: backend},
                ),
                patch.object(cmd, "openmp_flags", side_effect=OSError("missing compiler")),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                cmd.build_extensions()

    def test_invalid_prefix(self):
        _, cmd = self.command()
        with (
            patch.dict(os.environ, {"OMP_PREFIX": "/nonexistent/msda-test-prefix"}),
            self.assertRaisesRegex(RuntimeError, "include/omp.h"),
        ):
            cmd.openmp_flags()
