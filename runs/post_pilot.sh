#!/bin/bash
# After the branch pilot queue is empty: re-measure the freeze table (memory fix) and the 12 GB feasibility knobs.
source /home/yeho/miniforge3/etc/profile.d/conda.sh && conda activate ncrl_env
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=/home/yeho/RL/ncrl_light/runs/branch40k/queue.log
until grep -q "queue empty" $Q; do sleep 60; done
cd /home/yeho/RL/ncrl_light/tools
echo "$(date '+%F %T') start profile_update (freeze memory fix)" >> $Q
python -I profile_update.py --out ../results/profile_update.json > ../results/profile_update.log 2>&1
echo "$(date '+%F %T') start profile_variants" >> $Q
python -I profile_variants.py --out ../results/profile_variants.json > ../results/profile_variants.log 2>&1
echo "$(date '+%F %T') post_pilot done" >> $Q
