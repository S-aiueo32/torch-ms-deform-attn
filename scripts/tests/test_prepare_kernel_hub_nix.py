"""Validate artifact identity before a GPU is rented."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_kernel_hub_nix as nix


class PrepareNixTest(unittest.TestCase):
    def test_checked_archive_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = root / "record"
            record.mkdir()
            archive = root / "distribution.tar.gz"
            name = f"./{nix.VARIANT}/fixture.so"
            binary = b"native artifact fixture"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo(name)
                member.size = len(binary)
                bundle.addfile(member, io.BytesIO(binary))
            (record / "distribution.sha256").write_text(
                hashlib.sha256(archive.read_bytes()).hexdigest()
            )
            (record / "distribution-files.json").write_text(
                json.dumps({name: hashlib.sha256(binary).hexdigest()})
            )
            (record / "UPSTREAM.json").write_text(
                json.dumps({"revision": "a" * 40, "source_sha256": {}})
            )
            (record / "export-revision.txt").write_text("b" * 40)
            target = root / "prepared"
            for name in (
                "distribution.sha256",
                "distribution-files.json",
                "UPSTREAM.json",
                "export-revision.txt",
            ):
                (root / name).write_bytes((record / name).read_bytes())
            nix.prepare(root, target, "c" * 40, "123456")
            self.assertEqual((target / "build" / nix.VARIANT / "fixture.so").read_bytes(), binary)
            provenance = json.loads((target / "NIX_BUILD.json").read_text())
            self.assertEqual(provenance["build_run"], "123456")
            self.assertEqual(provenance["artifact_source_sha"], "a" * 40)
            self.assertEqual(
                provenance["files"], {"fixture.so": hashlib.sha256(binary).hexdigest()}
            )
            archive.write_bytes(archive.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                nix.prepare(root, root / "bad", "c" * 40, "123456")
            self.assertFalse((root / "bad").exists())

    def test_changed_sources_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "distribution.tar.gz").write_bytes(b"fixture")
            (root / "distribution.sha256").write_text(hashlib.sha256(b"fixture").hexdigest())
            (root / "UPSTREAM.json").write_text(
                json.dumps({"source_sha256": {"kernel-hub/flake.lock": "0" * 64}})
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                nix.prepare(root, root / "bad", "c" * 40, "123456")
            self.assertFalse((root / "bad").exists())

    def test_invalid_run_rejected_before_download_or_export(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(nix.subprocess, "run") as run:
            target = Path(tmp) / "artifact"
            for build_run in ("", "0", "-1", "12/34", "123\n", "latest"):
                with self.subTest(build_run=build_run):
                    with self.assertRaisesRegex(ValueError, "run ID"):
                        nix.download(target, build_run)
                    with self.assertRaisesRegex(ValueError, "run ID"):
                        nix.prepare(target, Path(tmp) / "prepared", "c" * 40, build_run)
            run.assert_not_called()
            self.assertFalse(target.exists())

    def test_download_actions_artifact(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(nix.subprocess, "run") as run:
            target = Path(tmp) / "artifact"
            nix.download(target, "123456")
            run.assert_called_once_with(
                [
                    "gh",
                    "run",
                    "download",
                    "123456",
                    "--name",
                    "kernel-hub-nix-123456",
                    "--dir",
                    str(target),
                ],
                check=True,
            )

    def test_release_archive_preparation_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = b"archived native artifact"
            member_name = f"./{nix.VARIANT}/fixture.so"
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
                member = tarfile.TarInfo(member_name)
                member.size = len(binary)
                bundle.addfile(member, io.BytesIO(binary))
            files = {
                "distribution.tar.gz": archive.getvalue(),
                "distribution.sha256": hashlib.sha256(archive.getvalue()).hexdigest(),
                "distribution-files.json": json.dumps(
                    {member_name: hashlib.sha256(binary).hexdigest()}
                ),
                "UPSTREAM.json": json.dumps({"revision": "a" * 40, "source_sha256": {}}),
                "export-revision.txt": "b" * 40,
            }

            def download_release(command, check):
                self.assertTrue(check)
                self.assertEqual(
                    command[:6],
                    [
                        "gh",
                        "release",
                        "download",
                        "validation/nix",
                        "--pattern",
                        "kernel-hub-nix-987654.zip",
                    ],
                )
                with zipfile.ZipFile(Path(command[-1]) / command[5], "w") as bundle:
                    for name, data in files.items():
                        bundle.writestr(name, data)

            artifact = root / "artifact"
            with patch.object(nix.subprocess, "run", side_effect=download_release):
                nix.download(artifact, "987654", "validation/nix")
            target = root / "prepared"
            nix.prepare(artifact, target, "c" * 40, "987654")
            self.assertEqual((target / "build" / nix.VARIANT / "fixture.so").read_bytes(), binary)
            self.assertEqual(
                json.loads((target / "NIX_BUILD.json").read_text())["build_run"], "987654"
            )
