"""Install the pinned Linux x86_64 uv binary before using any networked pip."""

import argparse
import hashlib
import io
import platform
import shutil
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.13.5"
SHA256 = "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f"
ARCHIVE = "uv-x86_64-unknown-linux-gnu"
URL = f"https://github.com/astral-sh/uv/releases/download/{VERSION}/{ARCHIVE}.tar.gz"


def install(data, destination):
    # The committed release checksum protects bootstrap even if the container's
    # old TLS root bundle accepts a compromised CA. Never disable TLS checking.
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("uv bootstrap checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        members = [bundle.getmember(f"{ARCHIVE}/{name}") for name in ("uv", "uvx")]
        if not all(member.isfile() for member in members):
            raise ValueError("Invalid uv release archive")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / Path(member.name).name
            with bundle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("This bootstrap targets the Linux x86_64 CUDA runners")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("Unexpected uv archive size")
    install(data, args.destination)
    print(f"Installed uv {VERSION}; release archive SHA-256 verified")


if __name__ == "__main__":
    main()
