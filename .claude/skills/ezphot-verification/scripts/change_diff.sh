#!/usr/bin/env bash
# change_diff.sh WS [--names]
# Print only the changes made in the ezphot checkout since snapshot.sh ran: the diff of tracked files against the
# snapshot commit, then files that became untracked after the snapshot (as /dev/null diffs).
# _workspace/ and .claude/ are left out. --names: list the changed paths only, one per line.
export GIT_OPTIONAL_LOCKS=0   # never rewrite .git/index of a checkout other sessions may use
set -eu
WS=${1:?usage: change_diff.sh WS [--names]}
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
base=$(cat "$WS/00_baseline.txt")
cd "$ROOT"
EXCL=(':(exclude)_workspace' ':(exclude).claude')
new=$(comm -13 "$WS/00_untracked.txt" \
      <(git ls-files --others --exclude-standard | grep -v -e '^_workspace/' -e '^\.claude/' | sort))

if [ "${2:-}" = "--names" ]; then
    git diff --name-only "$base" -- . "${EXCL[@]}"
    [ -z "$new" ] || printf '%s\n' "$new"
    exit 0
fi

git diff "$base" -- . "${EXCL[@]}"
[ -z "$new" ] || while IFS= read -r f; do git diff --no-index -- /dev/null "$f" || true; done <<< "$new"
