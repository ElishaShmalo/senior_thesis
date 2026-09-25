#!/bin/bash
# Run every check for the block_disorder code with one command, and save the full output
# to tests/last_test_run.log so it can be read back (by you, or by Claude through the
# connected folder) without copy-pasting.
#
#   bash stavskya_mc/block_disorder/tests/run_all_tests.sh
#
# What it runs:
#   1. test_dynamics.jl: kernel refactor, block_len, recorder, log grid (+ speed print)
#   2. every generator in local test mode (STAV_LOCAL_TEST=1), writing to
#      block_disorder/_local_test_output/ (cleared first; git-ignored)
#   3. check_local_test_output.py: Python loaders must find exactly the files Julia wrote
#   4. test_time_log_tools.py (only if pytest is installed)
# Override the interpreters with JULIA=/path/to/julia or PYTHON=/path/to/python if needed.
# Exit status is 0 only if everything passed.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BD=$(dirname "$HERE")
REPO=$(cd "$BD/../.." && pwd)
LOG="$HERE/last_test_run.log"
JULIA=${JULIA:-julia}
PYTHON=${PYTHON:-python3}

exec > >(tee "$LOG") 2>&1

echo "run started:  $(date)"
echo "repository:   $REPO  (commit $(git -C "$REPO" --no-optional-locks rev-parse --short HEAD 2>/dev/null || echo '?'),"\
     "$(git -C "$REPO" --no-optional-locks status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted paths)"
echo "julia:        $("$JULIA" --startup-file=no -e 'print(VERSION)' 2>/dev/null || echo 'NOT FOUND')"
echo "python:       $("$PYTHON" --version 2>&1)"

status=0
run () {
    local name=$1; shift
    echo; echo "===== $name"
    "$@"
    local rc=$?
    if [ $rc -eq 0 ]; then echo "===== $name: PASS"; else echo "===== $name: FAIL (exit $rc)"; status=1; fi
}

run "julia: test_dynamics.jl" "$JULIA" --startup-file=no --project="$REPO" "$HERE/test_dynamics.jl"

rm -rf "$BD/_local_test_output"
for g in "$BD"/generators/get_*.jl; do
    run "julia: local smoke run $(basename "$g")" env STAV_LOCAL_TEST=1 "$JULIA" --startup-file=no --project="$REPO" "$g"
done

run "python: check_local_test_output.py" "$PYTHON" "$HERE/check_local_test_output.py"

if "$PYTHON" -c "import pytest" 2>/dev/null; then
    run "python: test_time_log_tools.py" "$PYTHON" -m pytest -q "$HERE/test_time_log_tools.py"
else
    echo; echo "(pytest not installed for $PYTHON: skipped test_time_log_tools.py; 'pip install pytest' to include it)"
fi

echo
if [ $status -eq 0 ]; then echo "OVERALL: PASS"; else echo "OVERALL: FAIL"; fi
echo "finished:     $(date)   log: $LOG"
exit $status
