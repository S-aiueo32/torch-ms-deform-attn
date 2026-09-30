"""Check Nix evidence generation without Nix, a compiler, or GPU rental."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_kernel_hub_nix as nix
from record_kernel_hub_distribution import record

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_kernel_hub_nix.sh"


class DistributionManifestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "distribution.tar.gz"
        self.manifest = self.root / "distribution-files.json"

    def write_archive(self, members):
        with tarfile.open(self.archive, "w:gz") as bundle:
            for name, kind, data in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.REGTYPE:
                    member.size = len(data)
                bundle.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)

    def test_hashes_preserve_member_names_and_sorted_order(self):
        names = (f"./{nix.VARIANT}/z.so", f"./{nix.VARIANT}/nested/a.py")
        data = b"artifact" * 200_000  # Exercise incremental hashing across chunks.
        self.write_archive(
            [(".", tarfile.DIRTYPE, b""), (f"./{nix.VARIANT}", tarfile.DIRTYPE, b"")]
            + [(name, tarfile.REGTYPE, data) for name in names]
        )
        record(self.archive, self.manifest, nix.VARIANT)
        expected = {name: hashlib.sha256(data).hexdigest() for name in sorted(names)}
        self.assertEqual(self.manifest.read_text(), json.dumps(expected, indent=2) + "\n")

    def test_reject_invalid_archives_without_manifest(self):
        valid = (f"./{nix.VARIANT}/fixture.so", tarfile.REGTYPE, b"fixture")
        cases = [([], "Empty distribution")]
        for name in ("/absolute.so", "../outside.so", f"{nix.VARIANT}/../outside.so", "other/a"):
            cases.append(([valid, (name, tarfile.REGTYPE, b"")], "Invalid distribution path"))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            cases.append(
                ([valid, (f"{nix.VARIANT}/link", kind, b"")], "Unsupported archive member")
            )
        for duplicate in (valid[0], f"{nix.VARIANT}/fixture.so"):
            cases.append(
                ([valid, (duplicate, tarfile.REGTYPE, b"changed")], "Duplicate distribution path")
            )
        for members, error in cases:
            with self.subTest(members=members, error=error):
                self.write_archive(members)
                with self.assertRaisesRegex(ValueError, error):
                    record(self.archive, self.manifest, nix.VARIANT)
                self.assertFalse(self.manifest.exists())

    def test_nix_script_evidence_round_trip(self):
        # Only Nix is substituted. Execute the real export, Git commit, tar,
        # manifest CLI and consumer to catch broken producer/consumer wiring.
        fixture = self.root / "fixture" / nix.VARIANT
        fixture.mkdir(parents=True)
        binary = b"native artifact fixture"
        (fixture / "fixture.so").write_bytes(binary)
        commands = self.root / "bin"
        commands.mkdir()
        fake_nix = commands / "nix"
        fake_nix.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == '--version':\n"
            "    print('fixture Nix (no actual build)')\n"
            "elif sys.argv[1] == 'build':\n"
            "    target = sys.argv[sys.argv.index('--out-link') + 1]\n"
            "    Path(target).symlink_to(os.environ['NIX_FIXTURE'], target_is_directory=True)\n"
            "elif sys.argv[1] in ('eval', 'derivation', 'path-info'):\n"
            "    print('{}')\n"
            "else:\n"
            "    raise SystemExit('Unexpected Nix invocation')\n"
        )
        fake_nix.chmod(0o755)
        output = self.root / "build-output"
        environment = os.environ.copy()
        environment.update(
            PATH=f"{commands}{os.pathsep}{environment['PATH']}",
            NIX_FIXTURE=str(fixture.parent),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
        )
        result = subprocess.run(
            ["bash", str(SCRIPT), str(output)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = output / "evidence"
        self.assertEqual((evidence / "status.txt").read_text(), "passed\n")
        self.assertEqual(
            json.loads((evidence / "distribution-files.json").read_text()),
            {f"./{nix.VARIANT}/fixture.so": hashlib.sha256(binary).hexdigest()},
        )
        prepared = self.root / "prepared"
        nix.prepare(evidence, prepared, "c" * 40)
        self.assertEqual((prepared / "build" / nix.VARIANT / "fixture.so").read_bytes(), binary)
        self.assertEqual(
            json.loads((prepared / "NIX_BUILD.json").read_text())["files"],
            {"fixture.so": hashlib.sha256(binary).hexdigest()},
        )
