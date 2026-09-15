"""Offline failure-path tests for the Runpod controller; no GPU or credentials."""

import argparse
import base64
import copy
import importlib.util
import io
import subprocess
import tarfile
import tempfile
import unittest
import urllib.error
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, unquote, urlsplit

SPEC = importlib.util.spec_from_file_location(
    "runpod_ci", Path(__file__).resolve().parents[1] / "runpod_ci.py"
)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class FakeAPI:
    """A tiny in-memory provider with observable creation and deletion calls."""

    def __init__(self, pods=(), lose_create_response=False):
        self.pods = {pod["id"]: copy.deepcopy(pod) for pod in pods}
        self.calls = []
        self.lose_create_response = lose_create_response

    def quote(self, gpu, cap, count=1):
        return Decimal("0.27")

    def list_pods(self):
        return list(copy.deepcopy(self.pods).values())

    def get_pod(self, pod_id):
        if pod_id not in self.pods:
            raise ci.APIError("GET", 404)
        return copy.deepcopy(self.pods[pod_id])

    def request(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if method == "POST" and path == "/pods":
            pod = dict(payload, id="newpod", cost=0.27)
            self.pods[pod["id"]] = pod
            if self.lose_create_response:
                raise ci.APIError("POST", 503)
            return copy.deepcopy(pod)
        if method == "DELETE":
            self.pods.pop(path.rsplit("/", 1)[-1], None)
            return None
        raise AssertionError((method, path))


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(
            repository="example/torch-ms-deform-attn",
            repository_id="123",
            run_id="456",
            attempt="1",
            state_dir=self.root / "state",
            output_dir=self.root / "output",
            source=self.root / "source",
            gpu="A5000",
            max_hourly_usd=Decimal("0.50"),
            timeout_minutes=5,
            sanitizer="none",
            benchmark=False,
        )
        self.owner = ci.identity(self.args)
        self.created = (ci.utc_now() - timedelta(minutes=1)).isoformat()
        self.pod = {
            "id": "ownedpod",
            "name": ci.pod_name(self.owner),
            "env": dict(self.owner, CI_CREATED_AT=self.created),
            "cost": 0.27,
        }
        self.sleep = mock.patch.object(ci.time, "sleep").start()
        self.addCleanup(mock.patch.stopall)

    def save_state(self, pod_id="ownedpod"):
        ci.write_state(
            self.args.state_dir,
            {
                "owner": self.owner,
                "created_at": self.created,
                "pod_id": pod_id,
                "phase": "created",
            },
        )

    def test_cleanup_recovers_a_pod_without_local_state(self):
        api = FakeAPI([self.pod])
        ci.cleanup(api, self.args)
        self.assertEqual([call[:2] for call in api.calls], [("DELETE", "/pods/ownedpod")])
        self.assertFalse(api.pods)

    def test_cleanup_rejects_state_from_another_workflow(self):
        self.save_state()
        self.args.run_id = "457"
        api = FakeAPI([self.pod])
        with self.assertRaises(ci.ControllerError):
            ci.cleanup(api, self.args)
        self.assertFalse(api.calls)

    def test_cleanup_does_not_delete_a_same_name_pod_from_another_repository(self):
        self.pod["env"]["CI_REPOSITORY"] = "someone/else"
        api = FakeAPI([self.pod])
        ci.cleanup(api, self.args)
        self.assertIn("ownedpod", api.pods)
        self.assertFalse(api.calls)

    def test_cleanup_checks_metadata_again_before_delete(self):
        self.save_state()
        self.pod["env"]["CI_ATTEMPT"] = "2"
        api = FakeAPI([self.pod])
        with self.assertRaises(ci.ControllerError):
            ci.cleanup(api, self.args)
        self.assertFalse(api.calls)

    def test_cleanup_still_deletes_known_pod_when_listing_fails(self):
        self.save_state()
        api = FakeAPI([self.pod])
        api.list_pods = mock.Mock(side_effect=ci.APIError("GET", 503))
        # Unknown duplicate creation intents cannot be ruled out, so the
        # controller may report the listing error after deleting the known ID.
        try:
            ci.cleanup(api, self.args)
        except ci.ControllerError:
            pass
        self.assertNotIn("ownedpod", api.pods)
        self.assertIn(("DELETE", "/pods/ownedpod", None), api.calls)

    def test_cleanup_treats_missing_pod_as_already_deleted(self):
        self.save_state()
        api = FakeAPI()
        ci.cleanup(api, self.args)
        self.assertFalse(api.calls)
        self.assertEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_cleanup_failure_does_not_claim_verified_deletion(self):
        self.save_state()
        api = FakeAPI([self.pod])
        api.request = mock.Mock(side_effect=ci.APIError("DELETE", 503))
        with self.assertRaises(ci.ControllerError):
            ci.cleanup(api, self.args)
        self.assertNotEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_find_owned_skips_pod_deleted_between_list_and_get(self):
        api = FakeAPI()
        api.list_pods = mock.Mock(return_value=[self.pod])
        self.assertEqual(ci.find_owned(api, self.owner), [])

    def test_owned_pod_rejects_missing_malformed_and_future_timestamp(self):
        for value in (
            None,
            "invalid",
            "2026-01-01T00:00:00",
            (ci.utc_now() + timedelta(hours=1)).isoformat(),
        ):
            with self.subTest(value=value):
                self.pod["env"]["CI_CREATED_AT"] = value
                self.assertFalse(ci.owned_pod(self.pod, self.owner))

    def test_recovery_rejects_multiple_matching_creations(self):
        duplicate = copy.deepcopy(self.pod)
        duplicate["id"] = "duplicate"
        api = FakeAPI([self.pod, duplicate])
        with self.assertRaises(ci.ControllerError):
            ci.recover_create(api, self.owner, self.created)
        self.assertFalse(api.calls)

    def test_reaper_only_deletes_old_pods_owned_by_this_repository(self):
        old = copy.deepcopy(self.pod)
        old["id"] = "oldpod"
        old["env"]["CI_CREATED_AT"] = (ci.utc_now() - timedelta(hours=3)).isoformat()
        other = copy.deepcopy(old)
        other["id"] = "otherpod"
        other["env"]["CI_REPOSITORY"] = "another/project"
        malformed = copy.deepcopy(old)
        malformed["id"] = "malformedpod"
        malformed["env"]["CI_CREATED_AT"] = "invalid"
        api = FakeAPI([self.pod, old, other, malformed])
        ci.reap(api, self.args)
        self.assertEqual(set(api.pods), {"ownedpod", "otherpod", "malformedpod"})

    def prepare_run(self):
        mock.patch.object(ci.subprocess, "check_output", return_value="a" * 40).start()
        scripts = self.args.source / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "runpod_bootstrap.sh").write_text("#!/bin/bash\ntrue\n")

        def fake_subprocess(command, **kwargs):
            if command[0] == "git":
                target = next(
                    value.split("=", 1)[1] for value in command if value.startswith("--output=")
                )
                Path(target).write_bytes(b"fake source archive")
            return subprocess.CompletedProcess(command, 0)

        mock.patch.object(
            ci, "generate_keys", return_value=("ssh-ed25519 AAAA", "host-key-b64")
        ).start()
        mock.patch.object(ci.subprocess, "run", side_effect=fake_subprocess).start()

    def test_lost_creation_response_is_recovered_without_a_second_post(self):
        self.prepare_run()
        api = FakeAPI(lose_create_response=True)
        with mock.patch.object(
            ci, "wait_for_ssh", side_effect=ci.ControllerError("SSH unavailable")
        ):
            with self.assertRaisesRegex(ci.ControllerError, "SSH unavailable"):
                ci.run(api, self.args)
        self.assertEqual(sum(call[0] == "POST" for call in api.calls), 1)
        self.assertEqual(sum(call[0] == "DELETE" for call in api.calls), 1)
        self.assertFalse(api.pods)
        self.assertEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_definite_creation_rejections_preserve_status_and_still_cleanup(self):
        self.prepare_run()
        for status in (400, 401, 402, 403, 404, 422, 429):
            with self.subTest(status=status):
                self.args.state_dir = self.root / f"state-{status}"
                api = FakeAPI()
                failure = ci.APIError("POST /pods", status)
                api.request = mock.Mock(side_effect=failure)
                with (
                    mock.patch.object(ci, "recover_create") as recovery,
                    mock.patch.object(ci, "cleanup", wraps=ci.cleanup) as cleanup,
                ):
                    with self.assertRaises(ci.APIError) as caught:
                        ci.run(api, self.args)
                self.assertIs(caught.exception, failure)
                self.assertIn(f"HTTP {status}", str(caught.exception))
                api.request.assert_called_once()
                recovery.assert_not_called()
                cleanup.assert_called_once_with(api, self.args)
                self.assertFalse(api.pods)

    def test_failed_quote_never_creates_a_pod(self):
        self.prepare_run()
        api = FakeAPI()
        api.quote = mock.Mock(side_effect=ci.ControllerError("quote exceeds cap"))
        with self.assertRaisesRegex(ci.ControllerError, "quote exceeds cap"):
            ci.run(api, self.args)
        self.assertFalse(api.calls)
        self.assertIsNone(ci.read_state(self.args.state_dir))

    def test_expensive_created_pod_is_deleted_before_ssh_or_tests(self):
        self.prepare_run()
        api = FakeAPI()
        request = api.request

        def expensive_creation(method, path, payload=None):
            response = request(method, path, payload)
            if method == "POST":
                response["cost"] = 0.75
                api.pods[response["id"]]["cost"] = 0.75
            return response

        api.request = expensive_creation
        with mock.patch.object(ci, "wait_for_ssh") as wait:
            with self.assertRaisesRegex(ci.ControllerError, "exceeds"):
                ci.run(api, self.args)
        wait.assert_not_called()
        self.assertFalse(api.pods)

    def test_selected_torch_version_and_source_sha_reach_remote(self):
        self.prepare_run()
        for version in ("2.5.1", "2.7.1"):
            with self.subTest(torch_version=version):
                self.args.torch_version = version
                self.args.state_dir = self.root / f"state-{version}"
                api = FakeAPI()
                with (
                    mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
                    mock.patch.object(ci, "stream_command", return_value=0) as command,
                    mock.patch.object(ci, "collect_artifacts"),
                ):
                    ci.run(api, self.args)
                remote = command.call_args.args[0][-1]
                self.assertIn(f"CUDA_CHECKS_TORCH_VERSION={version} ", remote)
                self.assertIn(f"CUDA_CHECKS_SOURCE_SHA={'a' * 40} ", remote)
                payload = next(payload for method, _, payload in api.calls if method == "POST")
                self.assertIn(f"pytorch:{version}-", payload["image"])
                self.assertFalse(api.pods)
                self.assertEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_two_gpu_request_and_remote_requirement(self):
        self.prepare_run()
        self.args.gpu_count = 2
        self.args.max_hourly_usd = Decimal("1")
        api = FakeAPI()
        with (
            mock.patch.object(api, "quote", wraps=api.quote) as quote,
            mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
            mock.patch.object(ci, "stream_command", return_value=0) as command,
            mock.patch.object(ci, "collect_artifacts"),
        ):
            ci.run(api, self.args)
        quote.assert_called_once_with("A5000", Decimal("1"), count=2)
        payload = next(payload for method, _, payload in api.calls if method == "POST")
        self.assertEqual(payload["gpu"]["count"], 2)
        self.assertIn("CUDA_CHECKS_GPU_COUNT=2 ", command.call_args.args[0][-1])
        self.assertFalse(api.pods)

    def test_benchmark_collects_results_and_deletes_pod(self):
        self.prepare_run()
        self.args.benchmark = True
        api = FakeAPI()
        with (
            mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
            mock.patch.object(ci, "stream_command", return_value=0) as command,
            mock.patch.object(ci, "collect_artifacts") as collect,
        ):
            ci.run(api, self.args)
        self.assertIn(
            "run_cuda_checks.sh none /workspace/ci/results benchmark", command.call_args.args[0][-1]
        )
        collect.assert_called_once_with(["ssh"], self.args.output_dir)
        self.assertFalse(api.pods)
        self.assertEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_benchmark_failure_collects_logs_and_deletes_pod(self):
        self.prepare_run()
        self.args.benchmark = True
        api = FakeAPI()
        with (
            mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
            mock.patch.object(ci, "stream_command", return_value=1),
            mock.patch.object(ci, "collect_artifacts") as collect,
        ):
            with self.assertRaisesRegex(ci.ControllerError, "exit status 1"):
                ci.run(api, self.args)
        collect.assert_called_once_with(["ssh"], self.args.output_dir)
        self.assertFalse(api.pods)

    def test_cleanup_failure_after_passing_tests_fails_the_run(self):
        self.prepare_run()
        api = FakeAPI()
        request = api.request

        def failing_deletion(method, path, payload=None):
            if method == "DELETE":
                raise ci.APIError("DELETE", 503)
            return request(method, path, payload)

        api.request = failing_deletion
        with (
            mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
            mock.patch.object(ci, "stream_command", return_value=0),
            mock.patch.object(ci, "collect_artifacts"),
        ):
            with self.assertRaises(ci.ControllerError):
                ci.run(api, self.args)
        self.assertIn("newpod", api.pods)
        self.assertNotEqual(ci.read_state(self.args.state_dir)["phase"], "deleted")

    def test_cancellation_after_creation_still_deletes_pod(self):
        self.prepare_run()
        api = FakeAPI()
        with mock.patch.object(ci, "wait_for_ssh", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                ci.run(api, self.args)
        self.assertFalse(api.pods)
        self.assertIn(("DELETE", "/pods/newpod", None), api.calls)

    def test_test_and_artifact_failures_still_delete_pod(self):
        self.prepare_run()
        api = FakeAPI()
        with (
            mock.patch.object(ci, "wait_for_ssh", return_value=["ssh"]),
            mock.patch.object(ci, "stream_command", return_value=7),
            mock.patch.object(
                ci, "collect_artifacts", side_effect=ci.ControllerError("download failed")
            ),
        ):
            with self.assertRaisesRegex(ci.ControllerError, "exit status 7"):
                ci.run(api, self.args)
        self.assertFalse(api.pods)


class PriceAndTransportTest(unittest.TestCase):
    def setUp(self):
        self.api = ci.RunpodAPI("example-secret-key")
        self.cap = Decimal("0.50")

    def quote_response(self, **overrides):
        value = {
            "id": "NVIDIA RTX A5000",
            "secure": True,
            "availability": "HIGH",
            "price": {"secure": 0.27},
        }
        value.update(overrides)
        return value

    def test_catalog_quote_uses_secure_single_gpu_cuda_floor(self):
        with mock.patch.object(self.api, "request", return_value=self.quote_response()) as request:
            self.assertEqual(self.api.quote("A5000", self.cap), Decimal("0.27"))
        method, path = request.call_args.args[:2]
        self.assertEqual(method, "GET")
        parsed = urlsplit(path)
        self.assertTrue(unquote(parsed.path).endswith("/catalog/gpus/NVIDIA RTX A5000"))
        query = parse_qs(parsed.query)
        self.assertEqual(query["cloud"], ["SECURE"])
        self.assertEqual(query["product"], ["POD"])
        self.assertEqual(query["count"], ["1"])
        self.assertEqual(query["minCudaVersion"], ["12.4"])

    def test_two_gpu_quote_checks_total_rate(self):
        with mock.patch.object(self.api, "request", return_value=self.quote_response()) as request:
            self.assertEqual(self.api.quote("A5000", Decimal("1"), count=2), Decimal("0.54"))
            self.assertEqual(parse_qs(urlsplit(request.call_args.args[1]).query)["count"], ["2"])
            with self.assertRaisesRegex(ci.ControllerError, "exceeds"):
                self.api.quote("A5000", self.cap, count=2)

    def test_quote_rejects_unknown_capacity_wrong_gpu_and_excess_price(self):
        for response in (
            self.quote_response(availability="NONE"),
            self.quote_response(availability=None),
            self.quote_response(secure=False),
            self.quote_response(id="NVIDIA H100 PCIe"),
            self.quote_response(price={"secure": 0.51}),
            self.quote_response(price={}),
        ):
            with (
                self.subTest(response=response),
                mock.patch.object(self.api, "request", return_value=response),
            ):
                with self.assertRaises(ci.ControllerError):
                    self.api.quote("A5000", self.cap)

    def test_price_validation_rejects_nonfinite_and_negative_values(self):
        for value in (None, "NaN", "Infinity", "-0.01", True, "not-money"):
            with self.subTest(value=value):
                with self.assertRaises(ci.ControllerError):
                    ci.money(value)

    def test_created_price_missing_or_excess_does_not_pass(self):
        self.assertFalse(ci.validate_price({}, self.cap))
        self.assertTrue(ci.validate_price({"cost": 0.50}, self.cap))
        with self.assertRaises(ci.ControllerError):
            ci.validate_price({"cost": 0.500001}, self.cap)

    def test_http_failure_redacts_secret_and_does_not_retry_post(self):
        failure = urllib.error.HTTPError(
            "https://api.runpod.io/v2/pods",
            503,
            "example-secret-key",
            {},
            io.BytesIO(b"private response"),
        )
        self.addCleanup(failure.close)
        with mock.patch.object(ci.urllib.request, "urlopen", side_effect=failure) as opening:
            with self.assertRaises(ci.APIError) as caught:
                self.api.request("POST", "/pods", {"env": {"secret": "payload-secret"}})
        self.assertEqual(opening.call_count, 1)
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn("secret", str(caught.exception))
        request = opening.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer example-secret-key")
        self.assertNotIn("example-secret-key", request.full_url)

    def test_invalid_json_is_reported_as_api_failure(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"not json"
        with mock.patch.object(ci.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(ci.APIError):
                self.api.request("GET", "/pods")

    def test_api_key_is_removed_from_child_environment(self):
        with mock.patch.dict(
            ci.os.environ, {"RUNPOD_API_KEY": "example-secret-key", "PATH": "/usr/bin"}
        ):
            self.assertNotIn("RUNPOD_API_KEY", ci.child_environment())
            self.assertEqual(ci.child_environment()["PATH"], "/usr/bin")

    def test_ssh_uses_pinned_host_key_and_ci_user(self):
        command = ci.ssh_command(
            Path("/tmp/keys"),
            "unique-pod-name",
            {
                "ssh": {
                    "direct": {
                        "host": "203.0.113.10",
                        "port": 34567,
                        "username": "root",
                        "command": "untrusted command",
                    }
                },
            },
        )
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("HostKeyAlias=unique-pod-name", command)
        self.assertIn("UserKnownHostsFile=/tmp/keys/known_hosts", command)
        self.assertEqual(command[-1], "ci@203.0.113.10")
        self.assertNotIn("untrusted command", command)

    def test_workflow_gpu_identifiers_are_accepted(self):
        for alias, identifier in ci.GPU_IDS.items():
            with self.subTest(gpu=identifier):
                self.assertEqual(ci.parse_args(["check", "--gpu", identifier]).gpu, alias)

    def test_ssh_rejects_non_ip_hosts_and_invalid_port(self):
        for host, port in (("-oProxyCommand=touch /tmp/unsafe", 22), ("203.0.113.10", 65536)):
            with self.subTest(host=host, port=port):
                with self.assertRaises((ci.ControllerError, ValueError)):
                    ci.ssh_command(
                        Path("/tmp/keys"), "pod", {"ssh": {"direct": {"host": host, "port": port}}}
                    )

    def test_generated_known_hosts_uses_only_generated_host_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def keygen(command, **kwargs):
                key = Path(command[command.index("-f") + 1])
                key.write_text("private " + key.name)
                key.with_suffix(".pub").write_text("ssh-ed25519 public-" + key.name + " comment\n")

            with mock.patch.object(ci.subprocess, "run", side_effect=keygen):
                public, private = ci.generate_keys(directory, "known-pod")
            self.assertEqual(
                (directory / "known_hosts").read_text(), "known-pod ssh-ed25519 public-host\n"
            )
            self.assertEqual(base64.b64decode(private), b"private host")
            self.assertEqual(public, "ssh-ed25519 public-client comment")


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "artifacts.tar"
        self.destination = self.root / "output"

    def make_archive(self, name, kind=tarfile.REGTYPE, data=b"test log"):
        with tarfile.open(self.archive, "w") as bundle:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/tmp/other-file"
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            bundle.addfile(member, io.BytesIO(data) if member.size else None)

    def test_extracts_regular_allowed_file(self):
        self.make_archive("./cuda-checks.log")
        ci.extract_artifacts(self.archive, self.destination)
        self.assertEqual((self.destination / "cuda-checks.log").read_bytes(), b"test log")

    def test_rejects_paths_links_devices_and_executables(self):
        for name, kind in (
            ("../outside.log", tarfile.REGTYPE),
            ("/absolute.log", tarfile.REGTYPE),
            ("nested/output.log", tarfile.REGTYPE),
            ("link.log", tarfile.SYMTYPE),
            ("hardlink.log", tarfile.LNKTYPE),
            ("device.log", tarfile.CHRTYPE),
            ("script.sh", tarfile.REGTYPE),
            ("directory.log", tarfile.DIRTYPE),
        ):
            with self.subTest(name=name, kind=kind):
                self.make_archive(name, kind)
                with self.assertRaises(ci.ControllerError):
                    ci.extract_artifacts(self.archive, self.destination)
        self.assertFalse((self.root / "outside.log").exists())

    def test_refuses_to_follow_existing_destination_symlink(self):
        self.make_archive("cuda-checks.log")
        self.destination.mkdir()
        outside = self.root / "outside.log"
        outside.write_text("original")
        (self.destination / "cuda-checks.log").symlink_to(outside)
        with self.assertRaises((ci.ControllerError, FileExistsError)):
            ci.extract_artifacts(self.archive, self.destination)
        self.assertEqual(outside.read_text(), "original")

    def test_artifact_size_limit_rejects_before_writing(self):
        self.make_archive("cuda-checks.log", data=b"12345")
        with mock.patch.object(ci, "MAX_ARTIFACT_BYTES", 4):
            with self.assertRaises(ci.ControllerError):
                ci.extract_artifacts(self.archive, self.destination)
        self.assertFalse((self.destination / "cuda-checks.log").exists())


if __name__ == "__main__":
    unittest.main()
