#!/usr/bin/env bash
# Exercise the Pod's SSH bootstrap on a hosted Linux runner, without a GPU or API key.
set -euo pipefail
umask 077

bootstrap_test_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bootstrap_test_temp=$(mktemp -d "${TMPDIR:-/tmp}/runpod-bootstrap-test.XXXXXXXX")
bootstrap_test_container=

cleanup() {
    local bootstrap_test_status=$?
    trap - EXIT
    if [[ -n "$bootstrap_test_container" ]]; then
        if (( bootstrap_test_status != 0 )); then
            docker logs "$bootstrap_test_container" >&2 || true
        fi
        docker rm --force "$bootstrap_test_container" >/dev/null 2>&1 || true
    fi
    rm -rf -- "$bootstrap_test_temp"
    exit "$bootstrap_test_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ssh-keygen -q -t ed25519 -N '' -f "$bootstrap_test_temp/client_key"
ssh-keygen -q -t ed25519 -N '' -f "$bootstrap_test_temp/host_key"
{
    printf 'CI_SSH_PUBLIC_KEY=%s\n' "$(cat "$bootstrap_test_temp/client_key.pub")"
    printf 'CI_SSH_HOST_KEY_B64=%s\n' "$(base64 < "$bootstrap_test_temp/host_key" | tr -d '\n')"
    printf 'CI_MAX_SECONDS=300\n'
} > "$bootstrap_test_temp/container.env"

bootstrap_test_container=$(docker run --detach --rm \
    --publish 127.0.0.1::22 \
    --env-file "$bootstrap_test_temp/container.env" \
    --mount "type=bind,source=$bootstrap_test_directory/runpod_bootstrap.sh,target=/bootstrap.sh,readonly" \
    ubuntu:22.04 bash /bootstrap.sh)
bootstrap_test_address=$(docker port "$bootstrap_test_container" 22/tcp)
bootstrap_test_port=${bootstrap_test_address##*:}
case "$bootstrap_test_port" in
    ''|*[!0-9]*) printf 'Invalid published SSH port: %s\n' "$bootstrap_test_address" >&2; exit 1 ;;
esac

read -r bootstrap_test_key_type bootstrap_test_public_key _ < "$bootstrap_test_temp/host_key.pub"
printf '[127.0.0.1]:%s %s %s\n' \
    "$bootstrap_test_port" "$bootstrap_test_key_type" "$bootstrap_test_public_key" \
    > "$bootstrap_test_temp/known_hosts"
bootstrap_test_ssh=(ssh -F /dev/null -n -T \
    -i "$bootstrap_test_temp/client_key" -p "$bootstrap_test_port" \
    -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o GlobalKnownHostsFile=/dev/null -o UpdateHostKeys=no \
    -o ConnectTimeout=3 -o ConnectionAttempts=1 -o LogLevel=ERROR)

bootstrap_test_deadline=$((SECONDS + 180))
bootstrap_test_ready=0
while (( SECONDS < bootstrap_test_deadline )); do
    if "${bootstrap_test_ssh[@]}" \
        -o "UserKnownHostsFile=$bootstrap_test_temp/known_hosts" \
        ci@127.0.0.1 true 2> "$bootstrap_test_temp/ssh-error.log"; then
        bootstrap_test_ready=1
        break
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "$bootstrap_test_container" 2>/dev/null || true)" != true ]]; then
        printf 'Bootstrap container stopped before SSH became ready.\n' >&2
        cat "$bootstrap_test_temp/ssh-error.log" >&2
        exit 1
    fi
    sleep 2
done
if (( ! bootstrap_test_ready )); then
    printf 'SSH bootstrap did not become ready within 180 seconds.\n' >&2
    cat "$bootstrap_test_temp/ssh-error.log" >&2
    exit 1
fi

# Harmless stand-ins check executable resolution in the GPU image's tool paths.
docker exec "$bootstrap_test_container" bash -euo pipefail -c '
    install -d -m 0755 /opt/conda/bin /usr/local/cuda/bin
    printf "#!/bin/sh\nprintf conda-path-ok\n" > /opt/conda/bin/ci_python_probe
    printf "#!/bin/sh\nprintf cuda-path-ok\n" > /usr/local/cuda/bin/ci_cuda_probe
    chmod 0755 /opt/conda/bin/ci_python_probe /usr/local/cuda/bin/ci_cuda_probe
'
"${bootstrap_test_ssh[@]}" \
    -o "UserKnownHostsFile=$bootstrap_test_temp/known_hosts" ci@127.0.0.1 '
    set -eu
    test "$(id -u)" -ne 0
    test "$(id -un)" = ci
    test "$CUDA_HOME" = /usr/local/cuda
    test "$PATH" = /opt/conda/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    test "$LD_LIBRARY_PATH" = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
    test "$(command -v ci_python_probe)" = /opt/conda/bin/ci_python_probe
    test "$(command -v ci_cuda_probe)" = /usr/local/cuda/bin/ci_cuda_probe
    test "$(ci_python_probe)" = conda-path-ok
    test "$(ci_cuda_probe)" = cuda-path-ok
    test -w /workspace/ci/source
    test -w /workspace/ci/results
'

# The unrelated client public key deliberately supplies the wrong server identity.
read -r bootstrap_test_key_type bootstrap_test_public_key _ < "$bootstrap_test_temp/client_key.pub"
printf '[127.0.0.1]:%s %s %s\n' \
    "$bootstrap_test_port" "$bootstrap_test_key_type" "$bootstrap_test_public_key" \
    > "$bootstrap_test_temp/wrong_known_hosts"
if "${bootstrap_test_ssh[@]}" \
    -o "UserKnownHostsFile=$bootstrap_test_temp/wrong_known_hosts" \
    ci@127.0.0.1 true 2> "$bootstrap_test_temp/wrong-key-error.log"; then
    printf 'SSH unexpectedly accepted an incorrect pinned host key.\n' >&2
    exit 1
fi
case "$(cat "$bootstrap_test_temp/wrong-key-error.log")" in
    *'Host key verification failed'*|*'REMOTE HOST IDENTIFICATION HAS CHANGED'*) ;;
    *) cat "$bootstrap_test_temp/wrong-key-error.log" >&2; exit 1 ;;
esac

printf 'Runpod SSH bootstrap checks passed: non-root login, tool environment, and host-key pinning.\n'
