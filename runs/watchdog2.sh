#!/bin/bash
# Replaces watchdog.sh after the 2026-10-25 hold: finished = "queue empty" (after wm_every8); post-pilot profiling is
# on hold, so it is not watched. Problem checks are the same as before.
D=/home/yeho/RL/ncrl_light/runs/branch40k; Q=$D/queue.log
while :; do
  for name in $(grep -oE "end [a-z0-9_]+ exit" $Q | awk '{print $2}'); do
    [ -f $D/$name/branch_log.jsonl ] && ! grep -q '"kind": "end"' $D/$name/branch_log.jsonl && { echo "PROBLEM arm $name ended without finishing (see $D/$name.log)"; exit 1; }
  done
  grep -qE "exit=[1-9]" $Q && { echo "PROBLEM nonzero exit: $(grep -E 'exit=[1-9]' $Q | tail -1)"; exit 1; }
  oom=$(grep -l "torch.OutOfMemoryError" $D/*.log 2>/dev/null | head -1); [ -n "$oom" ] && { echo "PROBLEM OOM in $oom"; exit 1; }
  if grep -q "queue empty" $Q; then echo "DONE queue empty: $(grep -E 'end wm_every8' $Q | tail -1)"; exit 0; fi
  if ! pgrep -f "queue_runner.sh" >/dev/null; then echo "PROBLEM queue_runner died before queue empty"; exit 1; fi
  cur=$(grep -E "^[0-9-]+ [0-9:]+ start " $Q | tail -1 | awk '{print $4}')
  if [ -n "$cur" ] && ! grep -q "end $cur exit" $Q && [ -f $D/$cur/branch_log.jsonl ]; then
    age=$(( $(date +%s) - $(stat -c %Y $D/$cur/branch_log.jsonl) ))
    [ $age -gt 1800 ] && { echo "PROBLEM arm $cur: no progress for $((age/60)) min"; exit 1; }
  fi
  sleep 60
done
