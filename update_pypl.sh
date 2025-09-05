#!/bin/bash
set -e

# 1. Bump version in ezphot/__init__.py
# Assumes version is in a line like: __version__ = "0.2.1"
VERSION_LINE=$(grep '__version__' ezphot/__init__.py)
CURRENT_VERSION=$(echo $VERSION_LINE | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

echo "Updating version: $CURRENT_VERSION ? $NEW_VERSION"
sed -i "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" ezphot/__init__.py

# 2. Clean old build artifacts
rm -rf dist build *.egg-info

# 3. Build
python -m build

# 4. Upload to PyPI
python -m twine upload dist/*

echo "? Uploaded ezphot version $NEW_VERSION to PyPI"

