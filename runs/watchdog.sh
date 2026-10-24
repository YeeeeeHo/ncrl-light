#!/bin/bash
# Exits (printing one line) when the pilot queue + post-pilot profiling finish, or when something goes wrong:
#  - an arm ended without writing {"kind": "end"} to its branch_log (crash), or a nonzero exit in queue.log
#  - torch.OutOfMemoryError in an arm log
#  - queue_runner.sh died before "queue empty" / post_pilot.sh died before "post_pilot done"
#  - the running arm wrote no branch_log line for 30 min (hang)
D=/home/yeho/RL/ncrl_light/runs/branch40k; Q=$D/queue.log
while :; do
  if grep -q "post_pilot done" $Q; then
    bad=$(grep -l "Traceback" /home/yeho/RL/ncrl_light/results/profile_update.log /home/yeho/RL/ncrl_light/results/profile_variants.log 2>/dev/null)
    echo "DONE all arms + post-pilot profiling${bad:+ (Traceback in: $bad)}"; exit 0; fi
  for name in $(grep -oE "end [a-z0-9_]+" $Q | awk '{print $2}'); do
    [ -f $D/$name/branch_log.jsonl ] && ! grep -q '"kind": "end"' $D/$name/branch_log.jsonl && { echo "PROBLEM arm $name ended without finishing (see $D/$name.log)"; exit 1; }
  done
  grep -qE "exit=[1-9]" $Q && { echo "PROBLEM nonzero exit: $(grep -E 'exit=[1-9]' $Q | tail -1)"; exit 1; }
  oom=$(grep -l "torch.OutOfMemoryError" $D/*.log 2>/dev/null | head -1); [ -n "$oom" ] && { echo "PROBLEM OOM in $oom"; exit 1; }
  if ! grep -q "queue empty" $Q && ! pgrep -f "queue_runner.sh" >/dev/null; then echo "PROBLEM queue_runner died before queue empty"; exit 1; fi
  if ! grep -q "post_pilot done" $Q && ! pgrep -f "post_pilot.sh" >/dev/null; then echo "PROBLEM post_pilot.sh died"; exit 1; fi
  cur=$(grep "start " $Q | tail -1 | awk '{print $4}')
  if [ -n "$cur" ] && ! grep -q "end $cur" $Q && [ -f $D/$cur/branch_log.jsonl ]; then
    age=$(( $(date +%s) - $(stat -c %Y $D/$cur/branch_log.jsonl) ))
    [ $age -gt 1800 ] && { echo "PROBLEM arm $cur: no progress for $((age/60)) min"; exit 1; }
  fi
  sleep 60
done
