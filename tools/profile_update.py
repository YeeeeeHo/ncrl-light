"""Where does one NCRL fine-tuning update spend its compute? (paper size, RTX 4070)

Uses NCRL's own classes from ncrl_src (unmodified) and replays the exact op sequence of DreamerAgent.update:
  1. world-model (WM) update: encoder -> RSSM.observe (64-step scan) -> decoder -> KL + recon -> backward -> Adam
  2. reward-head update (online rows only, detached posteriors)
  3. imagination (16 steps from every posterior state) -> lambda returns -> actor loss (backprop through dynamics)
  4. critic update, slow-target update
Each phase is timed with CUDA events (warm-up first, median over iterations) and its peak memory is recorded.
FLOPs per module come from torch.utils.flop_counter.FlopCounterMode (forward + backward).

Weights: the released pretrained WM (dmcontrol.pt) is loaded; data are random tensors with the replay buffer's
shapes (cost does not depend on pixel values). Batch = 16 x 64 (12 online + 4 offline rows, as online_ratio 0.75).

Part B ("freeze table") times only the WM update when some modules are excluded from training, with a fresh Adam
over the trainable parameters only (NCRL's Optimizer would also weight-decay frozen params, see notes in output).

  python -I profile_update.py --out ../results/profile_update.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

NCRL = os.environ.get("NCRL_SRC", "/home/yeho/RL/ncrl_src")
CKPT = "/home/yeho/RL/ncrl_data/models/models/dmcontrol.pt"
PROBE = "/home/yeho/RL/ncrl_probe"


def build(act_dim=12):
    sys.path.insert(0, os.path.join(NCRL, "src"))
    from omegaconf import OmegaConf
    import envs
    from dreamer.dreamer import DreamerAgent
    cfg = OmegaConf.load(os.path.join(NCRL, "config/finetune.yaml"))
    cfg.ws.mode = "finetune"
    cfg.agent.rssm.deter, cfg.agent.rssm.hidden, cfg.agent.rssm.classes = 12288, 1536, 96
    cfg.agent.encoder.cnn_depth = 96
    torch.manual_seed(0)
    agent = DreamerAgent(cfg.agent, (3, 64, 64), act_dim, envs.MAX_ACTION_DIM, device="cuda")
    agent.load(CKPT, {"wm": True, "actor": False, "critic": False})
    return cfg, agent, envs.MAX_ACTION_DIM


def make_batch(B, T, A, act_dim, dev="cuda"):
    act = np.zeros((B, T, A), np.float32)
    act[..., :act_dim] = np.random.uniform(-1, 1, (B, T, act_dim))
    first = np.zeros((B, T), bool); first[:, 0] = True
    d = dict(observation=np.random.randint(0, 256, (B, T, 3, 64, 64), dtype=np.uint8), action=act,
             reward=np.random.rand(B, T, 1).astype(np.float32), discount=np.ones((B, T, 1), np.float32),
             is_first=first, is_terminal=np.zeros((B, T, 1), bool))
    return {k: torch.as_tensor(v, device=dev) for k, v in d.items()}


def count_params(agent):
    wm, ac = agent.wm, agent._task_behavior
    g = wm.rssm._gru
    n = lambda m: sum(p.numel() for p in m.parameters())
    rows = {
        "wm.encoder": n(wm.encoder),
        "wm.rssm._gru._in_deter": n(g._in_deter), "wm.rssm._gru._in_stoch": n(g._in_stoch),
        "wm.rssm._gru._in_action": n(g._in_action), "wm.rssm._gru._dynlayers": n(g._dynlayers),
        "wm.rssm._gru._dyngru": n(g._dyngru), "wm.rssm._prior": n(wm.rssm._prior),
        "wm.rssm._posterior": n(wm.rssm._posterior),
        "wm.decoder._fc_deter": n(wm.decoder._fc_deter), "wm.decoder._fc_stoch": n(wm.decoder._fc_stoch),
        "wm.decoder._conv_model": n(wm.decoder._conv_model),
        "wm.reward_model": n(wm.reward_model),
        "ac.actor": n(ac.actor), "ac.critic": n(ac.critic), "ac._target_critic": n(ac._target_critic),
    }
    rows["TOTAL wm (incl. reward)"] = n(wm)
    rows["TOTAL ac (incl. target)"] = n(ac)
    return rows


class Timer:
    def __init__(self):
        self.ev = []

    def mark(self, name):
        e = torch.cuda.Event(enable_timing=True); e.record(); self.ev.append((name, e))

    def result(self):
        torch.cuda.synchronize()
        out = {}
        for (n0, e0), (n1, e1) in zip(self.ev[:-1], self.ev[1:]):
            out[n1] = e0.elapsed_time(e1)
        return out


def wm_update_phased(agent, data, tm, trainable=None, opt=None):
    """Same ops as WorldModel.update (precision 32, so autocast/GradScaler are no-ops), split into timed phases.
    trainable=None -> NCRL's own optimizer over all WM params (the original). Otherwise only `trainable` params get
    requires_grad and `opt` (a torch Adam over them) is stepped; clip over trainable only, no weight decay on frozen."""
    import torch.nn.functional as F
    wm = agent.wm
    if trainable is None:
        wm.requires_grad_(True)
    else:
        wm.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
    tm.mark("start")
    embed = wm.encoder(data["observation"]); tm.mark("wm_fwd_encoder")
    post, prior = wm.rssm.observe(embed, data["action"], data["is_first"], None); tm.mark("wm_fwd_rssm_observe")
    kl, _ = wm.rssm.loss(post, prior); tm.mark("wm_fwd_kl")
    recon = wm.decoder(post["deter"], post["stoch"]); tm.mark("wm_fwd_decoder")
    rl = F.mse_loss(recon, data["observation"], reduction="none").sum([2, 3, 4]).mean([0, 1])
    loss = kl["dyn_loss"] * 1.0 + kl["repr_loss"] * 0.1 + rl; tm.mark("wm_fwd_recon_loss")
    if trainable is None or len(trainable) > 0:
        loss.backward(); tm.mark("wm_bwd")
        if trainable is None:
            o = wm.model_opt
            params = list(wm.parameters())
            torch.nn.utils.clip_grad_norm_(params, o._clip); tm.mark("wm_opt_clip")
            o._apply_weight_decay(params); tm.mark("wm_opt_weight_decay")
            o._opt.step(); tm.mark("wm_opt_adam")
            o._opt.zero_grad(); tm.mark("wm_opt_zero_grad")
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 100.0); tm.mark("wm_opt_clip")
            opt.step(); opt.zero_grad(); tm.mark("wm_opt_adam")
    wm.requires_grad_(False)
    return post


def ac_update_phased(agent, post, data, tm, starts=None):
    """Same ops as DreamerAgent.update after the WM step: reward-head update, then ActorCritic.update."""
    from dreamer.dreamer import stop_gradient
    wm, ac = agent.wm, agent._task_behavior
    n_on = 12
    tm.mark("start_ac")
    wm.update_reward(post["deter"][:n_on].detach(), post["stoch"][:n_on].detach(), data["reward"][:n_on])
    tm.mark("reward_head_update")
    start = {k: stop_gradient(v) for k, v in post.items()}
    if starts is not None:  # timing-only subset (flatten B*T, take first N)
        start = {k: v.reshape((-1,) + v.shape[2:])[:starts][None] for k, v in start.items()}
    reward_fn = lambda seq: agent.two_hot.decode(wm.reward_model(seq["deter"], seq["stoch"]))
    ac.actor.requires_grad_(True)
    seq = wm.imagine(ac.actor, start, data["is_terminal"], ac.cfg.imag_horizon); tm.mark("imag_rollout_fwd")
    seq["reward"] = reward_fn(seq)
    target, meta, _ = ac.target(seq); tm.mark("imag_reward_value_target_fwd")
    actor_loss, _ = ac.actor_loss(seq, target); tm.mark("actor_loss_fwd")
    o = ac.actor_opt
    actor_loss.backward(); tm.mark("actor_bwd_through_dynamics")
    p = list(ac.actor.parameters())
    torch.nn.utils.clip_grad_norm_(p, o._clip); o._apply_weight_decay(p); o._opt.step(); o._opt.zero_grad()
    tm.mark("actor_opt")
    ac.actor.requires_grad_(False)
    ac.critic.requires_grad_(True)
    seq = {k: stop_gradient(v) for k, v in seq.items()}
    critic_loss, _ = ac.critic_loss(seq, meta["target"]); tm.mark("critic_fwd")
    critic_loss.backward(); tm.mark("critic_bwd")
    o = ac.critic_opt; p = list(ac.critic.parameters())
    torch.nn.utils.clip_grad_norm_(p, o._clip); o._apply_weight_decay(p); o._opt.step(); o._opt.zero_grad()
    tm.mark("critic_opt")
    ac.critic.requires_grad_(False)
    ac.update_slow_target(); tm.mark("slow_target")


def run_timed(fn, iters, warmup):
    rows = []
    for i in range(warmup + iters):
        tm = Timer()
        torch.cuda.synchronize()
        fn(tm)
        r = tm.result()
        if i >= warmup:
            rows.append(r)
    keys = rows[0].keys()
    med = {k: statistics.median(r[k] for r in rows) for k in keys}
    return med, rows


def flops_by_module(fn, depth=3):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False, depth=depth)
    with fc:
        fn()
    out = {}
    for mod, ops in fc.get_flop_counts().items():
        out[mod] = sum(ops.values())
    return out, fc.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    cfg, agent, A = build()
    B, T, act_dim = 16, 64, 12
    res = {"setting": dict(B=B, T=T, imag_horizon=int(cfg.agent.imag_horizon), deter=12288, hidden=1536,
                           classes=96, cnn_depth=96, padded_act_dim=A, act_dim=act_dim, precision=32,
                           tf32_matmul=torch.backends.cuda.matmul.allow_tf32, torch=torch.__version__,
                           gpu=torch.cuda.get_device_name(0))}
    res["params"] = count_params(agent)
    print(json.dumps(res["params"], indent=1), flush=True)
    resident = torch.cuda.memory_allocated() / 2**30
    res["resident_after_build_GiB"] = resident

    # ---------- Part A1: original WM update, phased ----------
    torch.cuda.reset_peak_memory_stats()
    def prep():
        d = make_batch(B, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"]); return d
    med, _ = run_timed(lambda tm: wm_update_phased(agent, prep(), tm), a.iters, a.warmup)
    res["wm_update_ms"] = med
    res["wm_update_peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    print("WM update (ms):", {k: round(v, 1) for k, v in med.items()}, "sum", round(sum(med.values()), 1), flush=True)

    # ---------- Part A2: reward + imagination + actor-critic for several start counts ----------
    sys.path.insert(0, PROBE)
    res["ac_update_ms"] = {}
    res["ac_update_peak_GiB"] = {}
    data0 = make_batch(B, T, A, act_dim); data0["observation"] = agent.preprocess_img(data0["observation"])
    with torch.no_grad():
        emb = agent.wm.encoder(data0["observation"])
        post0, _ = agent.wm.rssm.observe(emb, data0["action"], data0["is_first"], None)
    import ckpt_patch
    from dreamer.wm import WorldModel
    orig_imagine = WorldModel.imagine
    for mode, starts in [("original", 128), ("original", 256), ("original", 512), ("original", 1024),
                         ("ckpt", 1024), ("slim", 1024)]:
        WorldModel.imagine = orig_imagine
        if mode == "ckpt":
            ckpt_patch.apply(ckpt=True)
        elif mode == "slim":
            ckpt_patch.apply(ckpt=False)
        key = f"{mode}_{starts}"
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated() / 2**30
        try:
            med, _ = run_timed(lambda tm: ac_update_phased(agent, post0, data0, tm, starts=None if starts == 1024 else starts),
                               a.iters, a.warmup)
            res["ac_update_ms"][key] = med
            res["ac_update_peak_GiB"][key] = torch.cuda.max_memory_allocated() / 2**30
            print(key, {k: round(v, 1) for k, v in med.items()}, "sum", round(sum(med.values()), 1),
                  "peak", round(res["ac_update_peak_GiB"][key], 2), "base", round(base, 2), flush=True)
        except torch.OutOfMemoryError as e:
            res["ac_update_ms"][key] = "OOM"
            res["ac_update_peak_GiB"][key] = f"OOM ({str(e).splitlines()[0][:160]})"
            print(key, "OOM", flush=True)
            agent.wm.requires_grad_(False); agent._task_behavior.requires_grad_(False)
            for o in (agent._task_behavior.actor_opt, agent._task_behavior.critic_opt):
                o._opt.zero_grad()
        torch.cuda.empty_cache()
    WorldModel.imagine = orig_imagine

    # ---------- Part A3: FLOPs per module ----------
    def wm_flops():
        d = make_batch(B, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"])
        tm = Timer()
        wm_update_phased(agent, d, tm)
    res["flops_wm_update"], res["flops_wm_update_total"] = flops_by_module(wm_flops, depth=4)
    S = 128
    def ac_flops():
        tm = Timer()
        ac_update_phased(agent, post0, data0, tm, starts=S)
    f, tot = flops_by_module(ac_flops, depth=4)
    res["flops_ac_update_128starts"], res["flops_ac_update_128starts_total"] = f, tot
    res["flops_note"] = "AC FLOPs measured with 128 start states; imagination/actor/critic FLOPs scale linearly with starts (x8 for the original 1024). Reward-head update is independent of starts."
    print("FLOPs WM update total (TFLOP):", res["flops_wm_update_total"] / 1e12, flush=True)
    print("FLOPs AC update @128 starts (TFLOP):", tot / 1e12, flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    # ---------- Part B: freeze table for the WM update ----------
    wm = agent.wm
    # free NCRL's own WM Adam state (2 x 285M floats) so a second Adam fits; Part B does not use it
    wm.model_opt._opt.state.clear()
    torch.cuda.empty_cache()
    FB = 8  # FLOPs counted at batch 8 and doubled (WM-update FLOPs are linear in batch); timing uses batch 16
    groups = {
        "encoder": list(wm.encoder.parameters()),
        "rssm_core": list(wm.rssm._gru.parameters()),
        "rssm_prior": list(wm.rssm._prior.parameters()),
        "rssm_post": list(wm.rssm._posterior.parameters()),
        "decoder": list(wm.decoder.parameters()),
    }
    configs = {
        "full (all WM)": ["encoder", "rssm_core", "rssm_prior", "rssm_post", "decoder"],
        "freeze encoder": ["rssm_core", "rssm_prior", "rssm_post", "decoder"],
        "freeze decoder": ["encoder", "rssm_core", "rssm_prior", "rssm_post"],
        "freeze enc+dec (train RSSM)": ["rssm_core", "rssm_prior", "rssm_post"],
        "freeze RSSM core (GRU)": ["encoder", "rssm_prior", "rssm_post", "decoder"],
        "freeze RSSM (all), train enc+dec": ["encoder", "decoder"],
        "train RSSM heads only (prior+post)": ["rssm_prior", "rssm_post"],
        "train decoder only": ["decoder"],
        "train encoder only": ["encoder"],
        "no WM update (fwd observe for starts)": [],
    }
    res["freeze_table"] = {}
    for name, gs in configs.items():
        params = [p for g in gs for p in groups[g]]
        opt = torch.optim.Adam(params, 1e-4, eps=1e-5) if params else None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if not params:
            def fn(tm):
                d = make_batch(B, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"])
                tm.mark("start")
                with torch.no_grad():
                    e = wm.encoder(d["observation"]); tm.mark("fwd_encoder")
                    wm.rssm.observe(e, d["action"], d["is_first"], None); tm.mark("fwd_observe")
        else:
            def fn(tm, params=params, opt=opt):
                d = make_batch(B, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"])
                wm_update_phased(agent, d, tm, trainable=params, opt=opt)
        med, _ = run_timed(fn, a.iters, a.warmup)
        peak = torch.cuda.max_memory_allocated() / 2**30
        n_tr = sum(p.numel() for p in params)
        if params:
            def ff(params=params, opt=opt):
                d = make_batch(FB, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"])
                wm_update_phased(agent, d, Timer(), trainable=params, opt=opt)
        else:
            def ff():
                d = make_batch(FB, T, A, act_dim); d["observation"] = agent.preprocess_img(d["observation"])
                with torch.no_grad():
                    e = wm.encoder(d["observation"]); wm.rssm.observe(e, d["action"], d["is_first"], None)
        _, ftot = flops_by_module(ff, depth=1)
        ftot = ftot * (B / FB)
        res["freeze_table"][name] = dict(ms=sum(med.values()), phases=med, peak_GiB=peak,
                                         trainable_params=n_tr, tflops=ftot / 1e12)
        print(f"{name:40s} {sum(med.values()):8.1f} ms  peak {peak:5.2f} GiB  trainable {n_tr/1e6:6.1f}M  {ftot/1e12:6.2f} TFLOP", flush=True)
        del opt, fn, ff  # the closures hold the Adam state; drop them before the next config's Adam is created
        torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
    wm.requires_grad_(False)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("saved", a.out)


if __name__ == "__main__":
    main()
