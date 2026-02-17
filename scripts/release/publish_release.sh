#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/release/publish_release.sh <version>"
  echo "Example: scripts/release/publish_release.sh 1.0.0"
  exit 1
fi

VERSION="$1"
TAG="v${VERSION#v}"
RELEASE_DIR="dist/releases/${TAG}"

if [[ ! -d "$RELEASE_DIR" ]]; then
  echo "Release directory not found: $RELEASE_DIR"
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install: https://cli.github.com/"
  exit 1
fi

# Create tag if missing locally
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  git tag "$TAG"
fi

git push origin "$TAG"

if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$RELEASE_DIR"/* --clobber
else
  gh release create "$TAG" "$RELEASE_DIR"/* --title "$TAG" --notes "RePath model release $TAG"
fi

echo "Published release $TAG"
