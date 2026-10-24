"""12 GB feasibility knobs: how batch size, sequence length and numeric precision change the time and peak memory of
one NCRL update (paper size). Same op sequence as DreamerAgent.update, via NCRL's own classes.

Variants (each built fresh, one at a time):
  WM update : B in {4, 8, 16, 32} at T=64;  T in {16, 32} at B=16;  fp32 / TF32 matmul / NCRL fp16 (precision=16:
              autocast fp16 + GradScaler, as NCRL's own config switch).
  AC update : 1024 starts with ckpt_patch (result-identical checkpointing): fp32 / TF32 / fp16;  and 256 random
              starts without checkpointing (fp32).
Precision changes are NOT result-identical (they change numerics); batch/sequence changes change the algorithm
(see hardware_feasibility.md). Timing: CUDA events, warm-up 3, median of 6.

  python -I profile_variants.py --out ../results/profile_variants.json
"""
import argparse
import gc
import json
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_update as P  # noqa: E402


def build(precision=32, B=16, T=64):
    sys.path.insert(0, os.path.join(P.NCRL, "src"))
    from omegaconf import OmegaConf
    import envs
    from dreamer.dreamer import DreamerAgent
    cfg = OmegaConf.load(os.path.join(P.NCRL, "config/finetune.yaml"))
    cfg.ws.mode = "finetune"
    cfg.agent.rssm.deter, cfg.agent.rssm.hidden, cfg.agent.rssm.classes = 12288, 1536, 96
    cfg.agent.encoder.cnn_depth = 96
    cfg.agent.precision = precision
    torch.manual_seed(0)
    agent = DreamerAgent(cfg.agent, (3, 64, 64), 12, envs.MAX_ACTION_DIM, device="cuda")
    agent.load(P.CKPT, {"wm": True, "actor": False, "critic": False})
    return agent, envs.MAX_ACTION_DIM


def timed(fn, iters=6, warmup=3):
    ts = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        if i >= warmup:
            ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    sys.path.insert(0, P.PROBE)
    res = {"wm": {}, "ac": {}}

    wm_variants = [("fp32", 32, False, 16, 64), ("fp32", 32, False, 4, 64), ("fp32", 32, False, 8, 64),
                   ("fp32", 32, False, 32, 64), ("fp32", 32, False, 16, 32), ("fp32", 32, False, 16, 16),
                   ("tf32", 32, True, 16, 64), ("fp16", 16, False, 16, 64)]
    for name, prec, tf32, B, T in wm_variants:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        agent, A = build(prec)
        key = f"{name}_B{B}_T{T}"
        torch.cuda.reset_peak_memory_stats()

        def step():
            d = P.make_batch(B, T, A, 12); d["observation"] = agent.preprocess_img(d["observation"])
            agent.update_wm(d, 0)
        try:
            ms = timed(step)
            res["wm"][key] = dict(ms=ms, peak_GiB=torch.cuda.max_memory_allocated() / 2**30, frames=B * T,
                                  ms_per_1k_frames=ms / (B * T) * 1000)
        except torch.OutOfMemoryError:
            res["wm"][key] = "OOM"
        print("WM", key, res["wm"][key], flush=True)
        del agent, step; gc.collect(); torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    import ckpt_patch
    from dreamer.wm import WorldModel
    orig = WorldModel.imagine
    for name, prec, tf32, starts, ck in [("fp32", 32, False, None, True), ("tf32", 32, True, None, True),
                                         ("fp16", 16, False, None, True), ("fp32", 32, False, 256, False)]:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        WorldModel.imagine = orig
        ckpt_patch.apply(ckpt=ck)
        agent, A = build(prec)
        d = P.make_batch(16, 64, A, 12); d["observation"] = agent.preprocess_img(d["observation"])
        with torch.no_grad():
            post, _ = agent.wm.rssm.observe(agent.wm.encoder(d["observation"]), d["action"], d["is_first"], None)
        key = f"{name}_{starts or 1024}starts_{'ckpt' if ck else 'nockpt'}"
        torch.cuda.reset_peak_memory_stats()
        reward_fn = lambda seq: agent.two_hot.decode(agent.wm.reward_model(seq["deter"], seq["stoch"]))
        try:
            def step():  # NCRL's own methods (autocast + GradScaler handled inside for precision 16)
                agent.wm.update_reward(post["deter"][:12].detach(), post["stoch"][:12].detach(), d["reward"][:12])
                start = {k: v.detach() for k, v in post.items()}
                if starts is not None:
                    idx = torch.randperm(16 * 64, device="cuda")[:starts]
                    start = {k: v.reshape((-1,) + v.shape[2:])[idx][None] for k, v in start.items()}
                agent._task_behavior.update(agent.wm, start, d["is_terminal"], reward_fn)
            ms = timed(step)
            res["ac"][key] = dict(ms=ms, peak_GiB=torch.cuda.max_memory_allocated() / 2**30)
        except torch.OutOfMemoryError:
            res["ac"][key] = "OOM"
        print("AC", key, res["ac"][key], flush=True)
        del agent, post, d, step, reward_fn; gc.collect(); torch.cuda.empty_cache()
        WorldModel.imagine = orig
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("saved", a.out)


if __name__ == "__main__":
    main()
