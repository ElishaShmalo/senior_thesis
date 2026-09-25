#!/bin/bash
# Run every check for the 2026-09 spin-chain changes with one command and save the output to
# heisen_spin_chain/tests/last_test_run.log (Claude can read it from the connected folder).
#
#   bash heisen_spin_chain/tests/run_spin_tests.sh
#
# 1. test_spin_refactor.jl: new functions vs the pre-2026-09 code (bit-identical where expected)
# 2. local runs (SPIN_LOCAL_TEST=1) of get_good_data_severalL.jl and get_sdiff_data_severalL0.jl,
#    launched from the repository root exactly like the submit_*.sh scripts
# 3. check_spin_local_output.py: files, names, t columns, and that the other 8 generator scripts
#    differ from those two only in their settings lines
# Override the interpreters with JULIA=... / PYTHON=... . Exit status 0 only if everything passed.

set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LOG="$HERE/last_test_run.log"
JULIA=${JULIA:-julia}
PYTHON=${PYTHON:-python3}

exec > >(tee "$LOG") 2>&1
echo "run started:  $(date)"
echo "repository:   $REPO  (commit $(git -C "$REPO" --no-optional-locks rev-parse --short HEAD 2>/dev/null || echo '?'))"
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

cd "$REPO" || exit 1
run "julia: test_spin_refactor.jl" "$JULIA" --startup-file=no --project="$REPO" "$HERE/test_spin_refactor.jl"

rm -rf "$HERE/_local_test_output"
for g in get_good_data_severalL.jl get_sdiff_data_severalL0.jl; do
    run "julia: local run $g" env SPIN_LOCAL_TEST=1 "$JULIA" --startup-file=no --project="$REPO" "heisen_spin_chain/lyapunov_exponents/$g"
done

run "python: check_spin_local_output.py" "$PYTHON" "$HERE/check_spin_local_output.py"

echo
if [ $status -eq 0 ]; then echo "OVERALL: PASS"; else echo "OVERALL: FAIL"; fi
echo "finished:     $(date)   log: $LOG"
exit $status
