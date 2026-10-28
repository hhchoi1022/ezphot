#!/usr/bin/env bash
# snapshot.sh WS
# Record the state of the ezphot checkout before the engineer starts, so that the engineer's changes can be told apart
# from the uncommitted work the tree already carries. Leaves the working tree, index and stash list alone:
# `git stash create` only writes a commit object (no `stash store`).
#   WS/00_baseline.txt     "<commit>"   (HEAD when the tree is clean)
#   WS/00_untracked.txt    untracked files at snapshot time (_workspace/ and .claude/ left out)
export GIT_OPTIONAL_LOCKS=0   # never rewrite .git/index of a checkout other sessions may use
set -eu
WS=${1:?usage: snapshot.sh WS}
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$WS"
b=$(git -C "$ROOT" stash create)
[ -n "$b" ] || b=$(git -C "$ROOT" rev-parse HEAD)
echo "$b" > "$WS/00_baseline.txt"
git -C "$ROOT" ls-files --others --exclude-standard | grep -v -e '^_workspace/' -e '^\.claude/' | sort \
    > "$WS/00_untracked.txt" || true
cat "$WS/00_baseline.txt"
