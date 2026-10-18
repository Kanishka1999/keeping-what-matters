#!/bin/bash
# Overnight laptop queue (2026-09-25). Waits for local_e1s.py to finish, then runs, one at a time:
#   1. gen_check.py          generation check of the first-token refusal measure (pre-registration section 5)
#   2. local_e1_e2.py        second seed, chat-like arm (section 4)
#   3. local_e1_e2.py        second seed, code arm
#   4. exp_e_natural.py      natural-image cross-modal test at 2B (section 6)
# Each step runs even if an earlier one fails; caffeinate keeps the Mac awake throughout.
cd "$(dirname "$0")"
WAIT_PID=${1:-73814}
echo "$(date '+%F %T') waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') start"
run() { local log=$1; shift; echo "$(date '+%F %T') >>> $* (log: $log)"; "$@" > "$log" 2>&1; echo "$(date '+%F %T') <<< exit $?"; }
run logs/gen_check.log python3 gen_check.py
run logs/local_e1_seed1_benign.log python3 local_e1_e2.py --arm benign --seed 1
run logs/local_e1_seed1_ood.log python3 local_e1_e2.py --arm ood --seed 1
run logs/E_natural_2B.log python3 exp_e_natural.py --model Qwen/Qwen3.5-2B --seed 0
echo "$(date '+%F %T') chain done"
