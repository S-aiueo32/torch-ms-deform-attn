"""Validate artifact identity before a GPU is rented."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

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
            nix.prepare(root, target, "c" * 40)
            self.assertEqual((target / "build" / nix.VARIANT / "fixture.so").read_bytes(), binary)
            provenance = json.loads((target / "NIX_BUILD.json").read_text())
            self.assertEqual(provenance["artifact_source_sha"], "a" * 40)
            self.assertEqual(
                provenance["files"], {"fixture.so": hashlib.sha256(binary).hexdigest()}
            )
            archive.write_bytes(archive.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                nix.prepare(root, root / "bad", "c" * 40)
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
                nix.prepare(root, root / "bad", "c" * 40)
            self.assertFalse((root / "bad").exists())
