"""Test bootstrap integrity and the controller's isolated TLS trust store."""

import hashlib
import io
import ssl
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap_uv
import runpod_ci


class BootstrapTest(unittest.TestCase):
    def test_only_checked_binaries_are_installed(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for name in ("uv", "uvx"):
                member = tarfile.TarInfo(f"{bootstrap_uv.ARCHIVE}/{name}")
                member.size = len(name)
                bundle.addfile(member, io.BytesIO(name.encode()))
        data = archive.getvalue()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "tools"
            with self.assertRaisesRegex(ValueError, "checksum"):
                bootstrap_uv.install(data, destination)
            self.assertFalse(destination.exists())
            with patch.object(bootstrap_uv, "SHA256", hashlib.sha256(data).hexdigest()):
                bootstrap_uv.install(data, destination)
            self.assertEqual({p.name for p in destination.iterdir()}, {"uv", "uvx"})
            for name in ("uv", "uvx"):
                self.assertEqual((destination / name).read_text(), name)
                self.assertEqual((destination / name).stat().st_mode & 0o777, 0o755)


class TLSContextTest(unittest.TestCase):
    def test_api_request_uses_the_verified_context(self):
        api = runpod_ci.RunpodAPI("fixture-secret")
        with patch.object(runpod_ci.urllib.request, "urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = b'{"pods": []}'
            self.assertEqual(api.list_pods(), [])
        self.assertIs(request.call_args.kwargs["context"], api.tls_context)

    def test_system_bundle_override_is_not_used(self):
        with patch.dict(runpod_ci.os.environ, {"SSL_CERT_FILE": "/nonexistent/unsafe.pem"}):
            context = runpod_ci.api_tls_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertGreater(len(context.get_ca_certs()), 0)
        self.assertTrue(all(len(value) == 64 for value in runpod_ci.BLOCKED_CA_SHA256))

    def test_unsafe_bundle_is_rejected_before_api_request(self):
        unsafe = b"unsafe root fixture"
        with (
            patch.object(runpod_ci.ssl, "create_default_context") as create,
            patch.object(runpod_ci, "BLOCKED_CA_SHA256", {hashlib.sha256(unsafe).hexdigest()}),
            patch.object(runpod_ci.urllib.request, "urlopen") as request,
        ):
            create.return_value.get_ca_certs.return_value = [unsafe]
            with self.assertRaisesRegex(runpod_ci.ControllerError, "Unsafe CA bundle"):
                runpod_ci.RunpodAPI("fixture-secret")
            request.assert_not_called()
