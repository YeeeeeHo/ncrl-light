#!/bin/bash
# Branch pilot from the Quadruped Walk 40k-frame state (seed-1 base run), 20 epochs (= +20k frames) per arm, seed 7.
# Reference arm with the original update already exists: ncrl_runs/branch_qw40k/full1024 (same snapshot, same seed).
source /home/yeho/miniforge3/etc/profile.d/conda.sh && conda activate ncrl_env
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/yeho/RL/ncrl_light/runs/branch40k
if [ -n "$1" ]; then  # optional: PID of an arm already running (started by the previous runner) to wait for first
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
  echo "$(date '+%F %T') end $2 (started by previous runner)" >> queue.log
fi
R=/home/yeho/RL/ncrl_runs/baseline/logs/finetune/dmc-quadruped-walk/09af53c751cc43f5b4f6d07d6a082f14_1
while :; do
  line=$(grep -v '^[[:space:]]*#' pending.txt | grep -v '^[[:space:]]*$' | head -n1)
  [ -z "$line" ] && break
  awk -v l="$line" '!d && $0==l {d=1; next} 1' pending.txt > pending.tmp && mv pending.tmp pending.txt
  name=${line%%|*}; args=${line#*|}
  echo "$(date '+%F %T') start $name | $args" >> queue.log
  python -I /home/yeho/RL/ncrl_light/tools/branch_selective.py $args -- --base_run $R --snapshot $R/snapshot_40000.full.pt --out $name --epochs 20 > $name.log 2>&1
  ec=$?
  echo "$(date '+%F %T') end $name exit=$ec" >> queue.log
done
echo "$(date '+%F %T') queue empty" >> queue.log
