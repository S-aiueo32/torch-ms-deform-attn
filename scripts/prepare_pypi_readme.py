"""Make repository-relative README links permanent for a PyPI release."""

import argparse
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

LINK = re.compile(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<target>[^)\s]+)(?P<suffix>\))")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def rewrite_links(markdown: str, repository: str, revision: str) -> tuple[str, int]:
    """Point relative inline links at the exact GitHub source revision."""
    if REPOSITORY.fullmatch(repository) is None:
        raise ValueError("repository must have the form OWNER/NAME")
    if REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be a full lowercase Git commit hash")

    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        target = match.group("target")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("#", "/")):
            return match.group(0)

        path = PurePosixPath(parsed.path)
        if ".." in path.parts:
            raise ValueError(f"README link escapes the repository root: {target}")

        count += 1
        permanent = f"https://github.com/{repository}/blob/{revision}/{target}"
        return f"{match.group('prefix')}{permanent}{match.group('suffix')}"

    return LINK.sub(replace, markdown), count


def prepare(path: Path, repository: str, revision: str) -> int:
    rewritten, count = rewrite_links(path.read_text(), repository, revision)
    if count == 0:
        raise ValueError(f"no relative Markdown links found in {path}")
    path.write_text(rewritten)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    changed = prepare(args.readme, args.repository, args.revision)
    print(f"Rewrote {changed} README links for {args.repository}@{args.revision}")
