#!/usr/bin/env bash
set -euo pipefail

# Run from the upstream checkout on an x86_64 Linux Nix host.
# This is a distribution build; GPU numerical validation is a separate gate.
output=${1:?Provide a new output directory}
mkdir "$output"
output=$(realpath "$output")
mkdir "$output/evidence"
exec > >(tee "$output/evidence/build.log") 2>&1
source_sha=$(git rev-parse HEAD)
source_date=$(git show -s --format=%cI HEAD)
printf '%s\n' "$source_sha" > "$output/evidence/source-revision.txt"
nix --version
df -h /
python3 kernel-hub/export.py "$output/source" --revision "$source_sha"
cd "$output/source"

# Bootstrap once when introducing the lock; retain it as a reviewable artifact.
# Subsequent builds use the committed lock and refuse dependency updates.
if [ ! -f flake.lock ]; then
  nix flake lock
fi
cp flake.lock UPSTREAM.json "$output/evidence/"
git init --quiet
git add .
GIT_AUTHOR_DATE="$source_date" GIT_COMMITTER_DATE="$source_date" \
  git -c user.name='Kernel export' -c user.email='kernel-export@localhost' \
  commit --quiet -m "Export $source_sha"
git rev-parse HEAD > "$output/evidence/export-revision.txt"
tar -czf "$output/evidence/export.tar.gz" .

variant=torch211-cxx11-cu126-x86_64-linux
target=".#redistributable.$variant"
printf '%s\n' "$variant" > "$output/evidence/variant.txt"
nix eval --no-update-lock-file --json .#archVariants > "$output/evidence/variants.json"
nix derivation show --no-update-lock-file "$target" > "$output/evidence/derivation.json"
nix build --no-update-lock-file -L "$target" --out-link "$output/result"
mkdir "$output/distribution"
cp -rL "$output/result/." "$output/distribution/"
test -d "$output/distribution/$variant"
find "$output/distribution" -name '*.so' -print -quit | grep -q .
tar -C "$output/distribution" -czf "$output/evidence/distribution.tar.gz" .
sha256sum "$output/evidence/distribution.tar.gz" > "$output/evidence/distribution.sha256"
nix path-info --json "$output/result" > "$output/evidence/store-path.json"
printf 'passed\n' > "$output/evidence/status.txt"
