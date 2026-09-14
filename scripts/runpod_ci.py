"""Run CUDA CI on one disposable Runpod Pod, then verify its deletion.

Only this controller reads RUNPOD_API_KEY. The GPU receives an archive of the
repository and temporary SSH keys, never a GitHub or Runpod API credential.
"""

import argparse
import base64
import ipaddress
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

REST_URL = "https://api.runpod.io/v2"
IMAGE = (
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:"
    "14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
)
GPU_IDS = {"A5000": "NVIDIA RTX A5000", "L4": "NVIDIA L4", "RTX4090": "NVIDIA GeForce RTX 4090"}
ARTIFACT_SUFFIXES = {".whl", ".log", ".txt", ".json", ".xml"}
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


class ControllerError(RuntimeError):
    pass


class APIError(ControllerError):
    def __init__(self, operation, status=None):
        self.status = status
        message = f"Runpod {operation} failed" + (f" (HTTP {status})" if status else "")
        if status == 402:
            message += (
                "; check prepaid credits and spending limits in Runpod Billing before retrying"
            )
        elif status in (401, 403):
            message += "; check RUNPOD_API_KEY and its permission for this operation"
        super().__init__(message)


def utc_now():
    return datetime.now(timezone.utc)


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("missing timezone")
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError) as error:
        raise ControllerError("Invalid Pod ownership timestamp") from error


def money(value):
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise InvalidOperation
        return result
    except (InvalidOperation, ValueError) as error:
        raise ControllerError("Runpod did not provide a valid hourly price") from error


class RunpodAPI:
    def __init__(self, key):
        if not key or any(character.isspace() for character in key):
            raise ControllerError("Set RUNPOD_API_KEY in the controller environment")
        self.key = key

    def request(self, method, path, payload=None):
        url = REST_URL + path
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "User-Agent": "torch-ms-deform-attn-ci",
            },
        )
        # In particular, POST /pods is sent exactly once. A lost response is
        # recovered by ownership metadata, not by repeating a rental request.
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise APIError(f"{method} response exceeded the size limit")
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raise APIError(method + " " + path, error.code) from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise APIError(method + " " + path) from None

    def list_pods(self):
        result = self.request("GET", "/pods")
        if not isinstance(result, dict) or not isinstance(result.get("pods"), list):
            raise ControllerError("Unexpected Runpod Pod-list format")
        return result["pods"]

    def get_pod(self, pod_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(pod_id)):
            raise ControllerError("Invalid Runpod Pod identifier")
        result = self.request("GET", "/pods/" + pod_id)
        if not isinstance(result, dict):
            raise ControllerError("Unexpected Runpod Pod format")
        return result

    def quote(self, gpu, cap):
        query = urllib.parse.urlencode(
            {
                "include": "AVAILABILITY",
                "product": "POD",
                "count": 1,
                "cloud": "SECURE",
                "minCudaVersion": "12.4",
            }
        )
        result = self.request(
            "GET", "/catalog/gpus/" + urllib.parse.quote(GPU_IDS[gpu], safe="") + "?" + query
        )
        if not isinstance(result, dict) or result.get("id") != GPU_IDS[gpu]:
            raise ControllerError("Runpod GPU-price query failed; no Pod was requested")
        if result.get("secure") is not True:
            raise ControllerError("Requested GPU is unavailable on Runpod Secure Cloud")
        stock = result.get("availability")
        if stock not in ("HIGH", "MEDIUM", "LOW"):
            raise ControllerError("No matching single-GPU capacity; no Pod was requested")
        prices = result.get("price")
        price = money(prices.get("secure") if isinstance(prices, dict) else None)
        if price > cap:
            raise ControllerError(f"GPU quote ${price}/hour exceeds the ${cap}/hour limit")
        print(f"GPU {gpu}: Secure Cloud catalog price ${price}/hour; capacity {stock}", flush=True)
        return price


def identity(args):
    return {
        "CI_REPOSITORY": args.repository,
        "CI_REPOSITORY_ID": str(args.repository_id),
        "CI_RUN_ID": str(args.run_id),
        "CI_ATTEMPT": str(args.attempt),
    }


def pod_name(owner):
    return "tda-gha-{CI_REPOSITORY_ID}-{CI_RUN_ID}-{CI_ATTEMPT}".format(**owner)


def owned_pod(pod, owner, created_at=None):
    metadata = pod.get("env")
    if not isinstance(metadata, dict) or pod.get("name") != pod_name(owner):
        return False
    if any(metadata.get(key) != value for key, value in owner.items()):
        return False
    try:
        created = timestamp(metadata.get("CI_CREATED_AT"))
    except ControllerError:
        return False
    if created > utc_now() + timedelta(minutes=5):
        return False
    return created_at is None or metadata.get("CI_CREATED_AT") == created_at


def find_owned(api, owner, created_at=None):
    found = []
    for item in api.list_pods():
        if not isinstance(item, dict) or item.get("name") != pod_name(owner):
            continue
        try:
            pod = api.get_pod(item.get("id", ""))
        except APIError as error:
            if error.status == 404:
                continue
            raise
        if owned_pod(pod, owner, created_at):
            found.append(pod)
    return found


def read_state(directory):
    path = directory / "state.json"
    if not path.exists():
        return None
    try:
        with path.open() as source:
            result = json.load(source)
    except (OSError, ValueError) as error:
        raise ControllerError("Cannot read controller state") from error
    if not isinstance(result, dict):
        raise ControllerError("Invalid controller state")
    return result


def write_state(directory, state):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(state, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, directory / "state.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def delete_owned(api, pod_id, owner, created_at=None):
    for attempt in range(6):
        try:
            pod = api.get_pod(pod_id)
        except APIError as error:
            if error.status == 404:
                print(f"Verified Pod {pod_id} is deleted", flush=True)
                return
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 8))
            continue
        if not owned_pod(pod, owner, created_at):
            raise ControllerError(
                "Refusing to delete a Pod whose ownership metadata does not match"
            )
        try:
            api.request("DELETE", "/pods/" + pod_id)
        except APIError as error:
            if error.status != 404 and attempt == 5:
                raise
        time.sleep(min(2**attempt, 8))
    try:
        api.get_pod(pod_id)
    except APIError as error:
        if error.status == 404:
            print(f"Verified Pod {pod_id} is deleted", flush=True)
            return
        raise
    raise ControllerError("Pod deletion could not be verified; run cleanup again")


def cleanup(api, args):
    owner = identity(args)
    state = read_state(args.state_dir)
    if state is not None and state.get("owner") != owner:
        raise ControllerError("Controller state belongs to a different workflow run")
    created = state.get("created_at") if state else None
    ids = set()
    if state and state.get("pod_id"):
        ids.add(state["pod_id"])
    discovery_error = None
    try:
        # Listing also recovers a successful create whose HTTP response was lost.
        for pod in find_owned(api, owner, created):
            ids.add(pod["id"])
    except ControllerError as error:
        discovery_error = error
    errors = []
    for pod_id in sorted(ids):
        try:
            delete_owned(api, pod_id, owner, created)
        except ControllerError as error:
            errors.append(str(error))
    if errors:
        raise ControllerError("; ".join(errors))
    if discovery_error is not None and not ids:
        raise discovery_error
    if discovery_error is not None:
        print("Pod listing was unavailable; deletion of the recorded Pod was verified", flush=True)
    if state is not None:
        state["phase"] = "deleted" if ids else "not_found"
        write_state(args.state_dir, state)
    print(
        f"Cleanup complete for {pod_name(owner)} ({len(ids)} Pod identifiers checked)", flush=True
    )


def recover_create(api, owner, created_at):
    for attempt in range(7):
        try:
            found = find_owned(api, owner, created_at)
        except APIError:
            found = []
        if len(found) > 1:
            raise ControllerError("Multiple Pods match one creation intent; cleanup is required")
        if found:
            return found[0]
        if attempt < 6:
            time.sleep(5)
    raise ControllerError(
        "Pod creation response was lost and recovery found no Pod; cleanup will still run"
    )


def child_environment():
    result = os.environ.copy()
    result.pop("RUNPOD_API_KEY", None)
    return result


def generate_keys(directory, name):
    for key in ("client", "host"):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(directory / key)],
            check=True,
            env=child_environment(),
            timeout=30,
        )
    host_public = (directory / "host.pub").read_text().split()
    (directory / "known_hosts").write_text(name + " " + " ".join(host_public[:2]) + "\n")
    return (
        (directory / "client.pub").read_text().strip(),
        base64.b64encode((directory / "host").read_bytes()).decode(),
    )


def ssh_command(directory, name, pod):
    direct = pod["ssh"]["direct"]
    address = str(ipaddress.ip_address(direct["host"]))
    port = int(direct["port"])
    if not 1 <= port <= 65535:
        raise ControllerError("Invalid Runpod SSH port")
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(directory / "client"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + str(directory / "known_hosts"),
        "-o",
        "HostKeyAlias=" + name,
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
        "ci@" + address,
    ]


def validate_price(pod, cap):
    if pod.get("cost") is None:
        return False
    actual = money(pod["cost"])
    if actual > cap:
        raise ControllerError(f"Created Pod price ${actual}/hour exceeds ${cap}/hour; deleting it")
    return True


def wait_for_ssh(api, args, state, deadline):
    ssh_deadline = min(deadline, time.monotonic() + 15 * 60)
    price_deadline = time.monotonic() + 90
    while time.monotonic() < ssh_deadline:
        pod = api.get_pod(state["pod_id"])
        if not owned_pod(pod, state["owner"], state["created_at"]):
            raise ControllerError("Created Pod ownership metadata does not match")
        if pod.get("status") in ("ERROR", "EXITED", "TERMINATED"):
            raise ControllerError("GPU Pod exited before SSH became available")
        if not validate_price(pod, args.max_hourly_usd):
            if time.monotonic() >= price_deadline:
                raise ControllerError("Created Pod hourly price is unavailable; deleting it")
            time.sleep(5)
            continue
        try:
            command = ssh_command(args.state_dir, pod_name(state["owner"]), pod)
        except (KeyError, ValueError, TypeError):
            time.sleep(10)
            continue
        try:
            result = subprocess.run(
                command + ["test -d /workspace/ci/source && test -d /workspace/ci/results"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_environment(),
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            time.sleep(5)
            continue
        if result.returncode == 0:
            print("Pinned SSH host key verified; GPU host is ready", flush=True)
            return command
        time.sleep(10)
    raise ControllerError("GPU SSH readiness timed out")


def stream_command(command, output, deadline, tee=False, limit=None, merge_stderr=True):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else None,
        env=child_environment(),
        start_new_session=True,
    )
    total = 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise ControllerError("Remote command exceeded the controller deadline")
                for key, _ in selector.select(timeout=1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if limit is not None and total > limit:
                        raise ControllerError("Remote output exceeded the artifact size limit")
                    output.write(chunk)
                    output.flush()
                    if tee:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
        return process.wait(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        process.stdout.close()


def extract_artifacts(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(archive, mode="r:") as bundle:
        members = bundle.getmembers()
        if len(members) > 100:
            raise ControllerError("Too many remote artifacts")
        for member in members:
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if (
                not member.isfile()
                or len(path.parts) != 1
                or path.name != name
                or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,200}", name)
                or (path.suffix not in ARTIFACT_SUFFIXES and not name.endswith(".tar.gz"))
            ):
                raise ControllerError("Unsafe or unexpected remote artifact")
            total += member.size
            if member.size < 0 or total > MAX_ARTIFACT_BYTES:
                raise ControllerError("Remote artifacts exceed the size limit")
            target = destination / name
            with bundle.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def collect_artifacts(ssh, directory):
    command = (
        "cd /workspace/ci/results && find . -maxdepth 1 -type f "
        "\\( -name '*.whl' -o -name '*.tar.gz' -o -name '*.log' -o -name '*.txt' -o -name '*.json' -o -name '*.xml' \\) "
        "-print0 | tar --null -T - -cf -"
    )
    archive = directory / "artifacts.tar"
    with archive.open("wb") as output:
        result = stream_command(
            ssh + ["bash -o pipefail -c " + shlex.quote(command)],
            output,
            time.monotonic() + 120,
            limit=MAX_ARTIFACT_BYTES,
            merge_stderr=False,
        )
    if result:
        raise ControllerError("Could not download CUDA artifacts")
    extract_artifacts(archive, directory / "artifacts")
    archive.unlink()


def run(api, args):
    deadline = time.monotonic() + args.timeout_minutes * 60
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.state_dir.chmod(0o700)
    if read_state(args.state_dir) is not None or any(args.state_dir.iterdir()):
        raise ControllerError(
            "Use a new empty state directory; run cleanup for a previous creation intent"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    owner = identity(args)
    bootstrap = args.source / "scripts/runpod_bootstrap.sh"
    if not bootstrap.is_file():
        raise ControllerError("Missing scripts/runpod_bootstrap.sh in the source checkout")
    api.quote(args.gpu, args.max_hourly_usd)
    public, host_private = generate_keys(args.state_dir, pod_name(owner))
    archive = args.state_dir / "source.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "archive",
            "--format=tar",
            "--output=" + str(archive),
            "HEAD",
        ],
        check=True,
        env=child_environment(),
        timeout=60,
    )
    created = utc_now().isoformat()
    state = {"owner": owner, "created_at": created, "phase": "intent", "pod_id": None}
    write_state(args.state_dir, state)
    payload = {
        "name": pod_name(owner),
        "image": IMAGE,
        "disk": 40,
        "gpu": {
            "id": GPU_IDS[args.gpu],
            "count": 1,
            "minRamPerGpu": 16,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.4",
        },
        "cloud": "SECURE",
        "ports": ["22/tcp"],
        "startSsh": False,
        "startJupyter": False,
        "args": "/bin/bash -c " + shlex.quote(bootstrap.read_text()),
        "env": dict(
            owner,
            CI_CREATED_AT=created,
            CI_SSH_PUBLIC_KEY=public,
            CI_SSH_HOST_KEY_B64=host_private,
            CI_MAX_SECONDS=str(args.timeout_minutes * 60),
        ),
    }
    ssh = None
    error = None
    try:
        print(f"Requesting one {args.gpu} Pod (limit ${args.max_hourly_usd}/hour)", flush=True)
        try:
            pod = api.request("POST", "/pods", payload)
        except APIError as failure:
            # Definite request rejection (including quota/auth/rate-limit errors)
            # is not a lost creation response. Preserve its HTTP status; the outer
            # finally still performs ownership-scoped cleanup. HTTP 408 remains
            # ambiguous, like a connection timeout or server-side failure.
            if (
                failure.status is not None
                and failure.status != 408
                and not 500 <= failure.status < 600
            ):
                raise
            pod = recover_create(api, owner, created)
        if not isinstance(pod, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(pod.get("id", ""))):
            pod = recover_create(api, owner, created)
        state.update(pod_id=pod["id"], phase="created")
        write_state(args.state_dir, state)
        validate_price(pod, args.max_hourly_usd)
        ssh = wait_for_ssh(api, args, state, deadline)
        with archive.open("rb") as source:
            subprocess.run(
                ssh + ["tar -xf - -C /workspace/ci/source"],
                stdin=source,
                check=True,
                env=child_environment(),
                timeout=60,
            )
        remaining = int(deadline - time.monotonic())
        if remaining <= 60:
            raise ControllerError("Insufficient controller time remains for CUDA checks")
        remote = (
            "cd /workspace/ci/source && timeout --signal=TERM --kill-after=30s "
            f"{remaining - 30}s bash scripts/run_cuda_checks.sh {args.sanitizer} /workspace/ci/results"
            + (" benchmark" if args.benchmark else "")
        )
        with (args.output_dir / "controller.log").open("wb") as output:
            result = stream_command(ssh + [remote], output, deadline, tee=True)
        if result:
            raise ControllerError(f"CUDA checks failed with exit status {result}")
    except BaseException as caught:
        error = caught
    finally:
        if ssh is not None:
            try:
                collect_artifacts(ssh, args.output_dir)
            except Exception as caught:
                print(
                    "Artifact collection failed; cleanup will continue", file=sys.stderr, flush=True
                )
                if error is None:
                    error = caught
        try:
            cleanup(api, args)
        except Exception as caught:
            print(
                "Pod cleanup failed; the separate cleanup job must retry",
                file=sys.stderr,
                flush=True,
            )
            # A cleanup failure always makes the job fail, even after passing tests.
            if error is None:
                error = caught
    if error is not None:
        raise error
    print("CUDA checks and verified Pod cleanup completed", flush=True)


def reap(api, args):
    cutoff = utc_now() - timedelta(hours=2)
    failures = 0
    for item in api.list_pods():
        if not isinstance(item, dict) or not str(item.get("name", "")).startswith(
            f"tda-gha-{args.repository_id}-"
        ):
            continue
        try:
            pod = api.get_pod(item.get("id", ""))
        except APIError as error:
            if error.status == 404:
                continue
            failures += 1
            continue
        metadata = pod.get("env") or {}
        if not isinstance(metadata, dict):
            continue
        owner = {
            key: metadata.get(key)
            for key in ("CI_REPOSITORY", "CI_REPOSITORY_ID", "CI_RUN_ID", "CI_ATTEMPT")
        }
        if (
            owner["CI_REPOSITORY"] != args.repository
            or owner["CI_REPOSITORY_ID"] != str(args.repository_id)
            or not all(
                re.fullmatch(r"[1-9][0-9]*", str(owner[key] or ""))
                for key in ("CI_RUN_ID", "CI_ATTEMPT")
            )
            or not owned_pod(pod, owner)
        ):
            continue
        if timestamp(metadata["CI_CREATED_AT"]) > cutoff:
            continue
        try:
            delete_owned(api, pod["id"], owner, metadata["CI_CREATED_AT"])
        except ControllerError:
            failures += 1
    if failures:
        raise ControllerError(f"Could not delete {failures} stale owned Pods")
    print("Stale owned-Pod cleanup completed", flush=True)


def positive_id(value):
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("must be a positive numeric identifier")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Rent one GPU, test, collect artifacts, and delete it"
    )
    cleanup_parser = commands.add_parser(
        "cleanup", help="Delete Pods belonging to one workflow attempt"
    )
    reap_parser = commands.add_parser(
        "reap", help="Delete this repository's owned Pods older than two hours"
    )
    check_parser = commands.add_parser(
        "check", help="Read-only API authentication and GPU-price check"
    )
    for subparser in (run_parser, cleanup_parser, reap_parser):
        subparser.add_argument("--repository", required=True)
        subparser.add_argument("--repository-id", type=positive_id, required=True)
    for subparser in (run_parser, cleanup_parser):
        subparser.add_argument("--run-id", type=positive_id, required=True)
        subparser.add_argument("--attempt", type=positive_id, required=True)
        subparser.add_argument("--state-dir", type=Path, required=True)
    for subparser in (run_parser, check_parser):
        subparser.add_argument(
            "--gpu", choices=list(GPU_IDS) + list(GPU_IDS.values()), default="A5000"
        )
        subparser.add_argument("--max-hourly-usd", type=money, default=Decimal("0.50"))
    run_parser.add_argument("--source", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--sanitizer", choices=("none", "memcheck", "racecheck", "synccheck"), default="none"
    )
    run_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the installed CUDA wheel after correctness tests",
    )
    run_parser.add_argument("--timeout-minutes", type=int, default=45)
    args = parser.parse_args(argv)
    if hasattr(args, "gpu") and args.gpu not in GPU_IDS:
        args.gpu = next(alias for alias, identifier in GPU_IDS.items() if identifier == args.gpu)
    if hasattr(args, "repository") and not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository
    ):
        parser.error("repository must be OWNER/REPOSITORY")
    if hasattr(args, "timeout_minutes") and not 5 <= args.timeout_minutes <= 60:
        parser.error("timeout-minutes must be between 5 and 60")
    if hasattr(args, "max_hourly_usd") and args.max_hourly_usd <= 0:
        parser.error("max-hourly-usd must be positive")
    for field in ("state_dir", "output_dir", "source"):
        if hasattr(args, field):
            setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.command == "run" and (
        args.state_dir == args.output_dir
        or args.state_dir in args.output_dir.parents
        or args.output_dir in args.state_dir.parents
    ):
        parser.error("state-dir and output-dir must be separate, non-nested directories")
    return args


def main(argv=None):
    args = parse_args(argv)
    api = RunpodAPI(os.environ.get("RUNPOD_API_KEY", ""))

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Controller interrupted; attempting cleanup")

    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "run":
        run(api, args)
    elif args.command == "cleanup":
        cleanup(api, args)
    elif args.command == "reap":
        reap(api, args)
    else:
        api.list_pods()
        api.quote(args.gpu, args.max_hourly_usd)
        print("Runpod read-only authentication check passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Runpod CI interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (ControllerError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
