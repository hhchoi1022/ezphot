#!/usr/bin/env bash
# post_run_check.sh [--pre] T [INPUT_DIR ...]
# post_run_check.sh --real-pre STATE DIR ... | --real STATE DIR ...
#   real-archive mode (approved runs with the real HOME, ezphot-photometry): products are written into the real
#   archive on purpose, so only disappearance is checked. --real-pre lists the *.fits under DIR into STATE/real_inputs.txt
#   (STATE must lie under _workspace/); --real prints every listed FITS that no longer exists and exits 1 if any.
# Integrity check around a run in the isolated HOME T. Read-only outside T.
#   --pre T DIR...   before the run: list the files of the input dirs (path, size, mtime) into T/input_files.txt.
#   T [DIR...]       after the run:
#     strict  files newer than T/marker in the real ezphot config, refdata, calibdata, mcalibdata, obsdata and the
#             given input dirs; and, if T/input_files.txt exists, files of the input dirs that disappeared or changed
#             since --pre (this is the only part that sees deletions). Any hit -> printed, exit 1.
#     catalog new files in the real CATALOG_DIR (skycatalog cache; the isolated HOME keeps it real on purpose).
#             Printed with a CATALOG prefix, exit code unchanged: report them.
#     info    files newer than the marker in the real ~/ezphot/log (other processes may write there). INFO prefix.
# T must lie under $HARNESS_ISO_BASE (default: BASE of tests/make_isolated_home.py).
export GIT_OPTIONAL_LOCKS=0   # never rewrite .git/index of a checkout other sessions may use
set -u
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
if [ "${1:-}" = "--real-pre" ] || [ "${1:-}" = "--real" ]; then
    mode=$1; STATE=${2:?usage: post_run_check.sh $mode STATE DIR ...}; shift 2
    case "$(readlink -m "$STATE")" in "$ROOT"/_workspace/*) ;; *) echo "refusing: $STATE is not under $ROOT/_workspace" >&2; exit 2 ;; esac
    if [ "$mode" = "--real-pre" ]; then
        mkdir -p "$STATE"
        for d in "$@"; do [ -d "$d" ] && find "$d" -type f -name '*.fits'; done | sort > "$STATE/real_inputs.txt"
        echo "$(wc -l < "$STATE/real_inputs.txt") FITS listed in $STATE/real_inputs.txt"; exit 0
    fi
    [ -f "$STATE/real_inputs.txt" ] || { echo "no $STATE/real_inputs.txt (run --real-pre first)" >&2; exit 2; }
    gone=$(while IFS= read -r f; do [ -e "$f" ] || echo "DELETED $f"; done < "$STATE/real_inputs.txt")
    [ -z "$gone" ] && exit 0
    printf '%s\n' "$gone"; exit 1
fi
PRE=0
[ "${1:-}" = "--pre" ] && { PRE=1; shift; }
T=${1:?usage: post_run_check.sh [--pre] T [INPUT_DIR ...]}
shift
# default: BASE in ezphot's tests/make_isolated_home.py (the script that makes T), so the two never disagree
mih="$ROOT/tests/make_isolated_home.py"
default_base=$(sed -n "s/^BASE = Path('\(.*\)').*/\1/p" "$mih" 2>/dev/null | head -1)
ISO_BASE=$(readlink -f "${HARNESS_ISO_BASE:-${default_base:?cannot read BASE from $mih; set HARNESS_ISO_BASE}}")
case "$(readlink -f "$T")" in
    "$ISO_BASE"/*) ;;
    *) echo "refusing: $T is not under $ISO_BASE" >&2; exit 2 ;;
esac
[ -f "$T/marker" ] || { echo "no marker in $T" >&2; exit 2; }

list_inputs() { for d in "$@"; do [ -d "$d" ] && find "$d" -type f -printf '%p\t%s\t%T@\n'; done | sort; }

if [ $PRE = 1 ]; then
    list_inputs "$@" > "$T/input_files.txt"
    echo "$(wc -l < "$T/input_files.txt") input files listed in $T/input_files.txt"
    exit 0
fi

REAL_HOME=${REAL_HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
EZ=$REAL_HOME/ezphot
CATDIR=$EZ/data/skycatalog/archive
real_cfg=$EZ/config/common/catalog.config
[ -f "$real_cfg" ] && CATDIR=$(grep -o '"CATALOG_DIR": *"[^"]*"' "$real_cfg" | sed 's/.*: *"\(.*\)"/\1/')
STRICT=("$EZ/config" "$EZ/data/refdata" "$EZ/data/calibdata" "$EZ/data/mcalibdata" "$EZ/data/obsdata" "$@")

newer() { for d in "$@"; do [ -e "$d" ] && find "$d" -newer "$T/marker" -print 2>/dev/null; done; }

info=$(newer "$EZ/log")
[ -z "$info" ] || { echo "INFO $(printf '%s\n' "$info" | wc -l) new files in $EZ/log (check whether any belong to this run):";
                    printf '%s\n' "$info" | head -20 | sed 's/^/INFO /'; }
cat_new=$(newer "$CATDIR")
[ -z "$cat_new" ] || printf '%s\n' "$cat_new" | sed 's/^/CATALOG /'

hits=$(newer "${STRICT[@]}")
if [ -f "$T/input_files.txt" ] && [ $# -gt 0 ]; then
    gone=$(comm -23 "$T/input_files.txt" <(list_inputs "$@") | cut -f1)
    [ -z "$gone" ] || hits=$(printf '%s\n%s' "$hits" "$(printf '%s\n' "$gone" | sed 's/^/CHANGED-OR-DELETED /')")
fi
hits=$(printf '%s\n' "$hits" | sed '/^$/d')
if [ -n "$hits" ]; then
    printf '%s\n' "$hits"
    exit 1
fi
exit 0
