#!/usr/bin/env bash
# Executed as the container entrypoint. Only one-job SSH credentials enter the Pod.
set -euo pipefail
umask 077

: "${CI_SSH_PUBLIC_KEY:?Missing temporary SSH public key}"
: "${CI_SSH_HOST_KEY_B64:?Missing temporary SSH host key}"
: "${CI_MAX_SECONDS:?Missing execution deadline}"
case "$CI_MAX_SECONDS" in
  ''|*[!0-9]*) exit 2 ;;
esac
if (( CI_MAX_SECONDS < 60 || CI_MAX_SECONDS > 7200 )); then
  echo 'Invalid Pod process deadline' >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
ci_packages=(openssh-server ca-certificates git build-essential python3-venv)
if [[ "${CI_WORKLOAD:-core}" == kernel-hub ]]; then
  ci_packages+=(pkg-config libssl-dev curl)
fi
apt-get install -y --no-install-recommends "${ci_packages[@]}"
useradd --create-home --shell /bin/bash ci
# A disabled password with public-key login; the SSH server rejects passwords.
usermod --password '*' ci
install -d -m 0755 /run/sshd /workspace
install -d -m 0700 -o ci -g ci /home/ci/.ssh /workspace/ci
install -d -m 0755 -o ci -g ci /workspace/ci/source /workspace/ci/results
printf '%s\n' "$CI_SSH_PUBLIC_KEY" > /home/ci/.ssh/authorized_keys
chmod 0600 /home/ci/.ssh/authorized_keys
chown ci:ci /home/ci/.ssh/authorized_keys
printf '%s' "$CI_SSH_HOST_KEY_B64" | base64 --decode > /etc/ssh/ci_host_key
chmod 0600 /etc/ssh/ci_host_key
unset CI_SSH_PUBLIC_KEY CI_SSH_HOST_KEY_B64

cat > /etc/ssh/sshd_config_ci <<'CONFIG'
Port 22
HostKey /etc/ssh/ci_host_key
PidFile /run/sshd_ci.pid
AuthorizedKeysFile .ssh/authorized_keys
AllowUsers ci
PermitRootLogin no
PasswordAuthentication no
PermitEmptyPasswords no
KbdInteractiveAuthentication no
UsePAM yes
AllowAgentForwarding no
AllowTcpForwarding no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no
PrintMotd no
LogLevel ERROR
Subsystem sftp internal-sftp
CONFIG

# Non-interactive SSH commands do not source the image's interactive shell setup.
cat > /etc/environment <<'ENVIRONMENT'
PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CUDA_HOME=/usr/local/cuda
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENVIRONMENT
/usr/sbin/sshd -t -f /etc/ssh/sshd_config_ci
echo 'CUDA worker SSH service ready'
# This bounds remote processes, not Runpod billing. The controller and recovery
# workflow delete the Pod through the provider API, including on test failures.
exec timeout --signal=TERM --kill-after=10 "$CI_MAX_SECONDS" \
  /usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_ci
