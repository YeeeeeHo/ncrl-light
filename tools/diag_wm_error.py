"""Experiment C, forward-only part: how wrong is the world model on new online data, which modules actually moved
during fine-tuning, and which of those moves matter for prediction / value?

No training. Uses the existing NCRL Quadruped Walk run (seed 1, ncrl_runs/baseline; unmodified NCRL updates with
the result-identical checkpointed imagination) and the released pretrained model:
  pre  = dmcontrol.pt (WM only)               ft20 = snapshot at 20k frames (~1.25k updates)
  ft40 = snapshot at 40k frames (~3.75k updates)
Data (64-step chunks, fixed sampling seed):
  on_bc   = online episodes 0-14   (collected by the BC prior during warm-up)
  on_rl   = online episodes 15-39  (RL + guidance; trained on by ft40)
  on_new  = online episodes 40-50  (collected after the 40k snapshot: unseen by ft20 and ft40)
  prior   = the 500 retrieved episodes (rehearsal data)
  walker / cheetah / cartpole / acrobot = pretraining pool of other embodiments (forgetting probe)
Metrics per chunk: recon (posterior pixel MSE, summed over pixels, as the training loss), kl (KL(post||prior),
no free-nats), openloop (16 observed frames, then 32 imagined steps with the logged actions; pixel MSE),
and for ft models: reward error on online data and value gap |V(open-loop prior) - V(posterior)|.
Part 2: per-module parameter drift (relative L2) pre -> ft20 -> ft40.
Part 3: module reversion: ft40 with ONE module group put back to its pretrained weights; the change in error on
on_new tells how much that group's fine-tuning matters.

  python -I diag_wm_error.py --out ../results/diag_wm_error.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

NCRL = os.environ.get("NCRL_SRC", "/home/yeho/RL/ncrl_src")
RUN = "/home/yeho/RL/ncrl_runs/baseline/logs/finetune/dmc-quadruped-walk/09af53c751cc43f5b4f6d07d6a082f14_1"
PRE = "/home/yeho/RL/ncrl_data/models/models/dmcontrol.pt"
POOL = "/home/yeho/RL/ncrl_data/ncrl_datasets/dmcontrol"
T, CTX, HOR = 64, 16, 32
DEV = os.environ.get("DEV", "cuda")


def ep_index(fn):
    return int(re.match(r".*-(\d+)-\d+\.npz$", fn).group(1))


def load_chunks(files, n, rng, A):
    obs, act, rew, first = [], [], [], []
    for _ in range(n):
        f = files[rng.integers(len(files))]
        d = np.load(f)
        L = len(d["action"])
        i = rng.integers(0, L - T + 1)
        o = d["observation"][i:i + T]; a = d["action"][i:i + T]; r = d["reward"][i:i + T]
        pa = np.zeros((T, A), np.float32); pa[:, :a.shape[1]] = a
        obs.append(o); act.append(pa); rew.append(r)
    fs = np.zeros((n, T), bool); fs[:, 0] = True
    return dict(observation=np.stack(obs), action=np.stack(act), reward=np.stack(rew), is_first=fs)


def build():
    sys.path.insert(0, os.path.join(NCRL, "src"))
    from omegaconf import OmegaConf
    import envs
    from dreamer.dreamer import DreamerAgent
    cfg = OmegaConf.load(os.path.join(NCRL, "config/finetune.yaml"))
    cfg.ws.mode = "finetune"
    cfg.agent.rssm.deter, cfg.agent.rssm.hidden, cfg.agent.rssm.classes = 12288, 1536, 96
    cfg.agent.encoder.cnn_depth = 96
    agent = DreamerAgent(cfg.agent, (3, 64, 64), 12, envs.MAX_ACTION_DIM, device=DEV)
    return agent, envs.MAX_ACTION_DIM


def load_weights(agent, which, pre_sd, snaps):
    if which == "pre":
        agent.wm.encoder.load_state_dict(pre_sd["wm"]["encoder"])
        agent.wm.rssm.load_state_dict(pre_sd["wm"]["rssm"])
        agent.wm.decoder.load_state_dict(pre_sd["wm"]["decoder"])
    else:
        s = snaps[which]
        agent.wm.encoder.load_state_dict(s["wm"]["encoder"])
        agent.wm.rssm.load_state_dict(s["wm"]["rssm"])
        agent.wm.decoder.load_state_dict(s["wm"]["decoder"])
        agent.wm.reward_model.load_state_dict(s["reward"])
        agent._task_behavior.critic.load_state_dict(s["critic"])


@torch.no_grad()
def evaluate(agent, data, has_heads, bs=16):
    wm = agent.wm
    out = {k: [] for k in ["recon", "kl", "openloop", "openloop_h1_8", "openloop_h25_32", "reward_err", "value_gap", "value_gap_h25_32"]}
    n = data["observation"].shape[0]
    for s in range(0, n, bs):
        d = {k: torch.as_tensor(v[s:s + bs], device=DEV) for k, v in data.items()}
        o = d["observation"].float() / 255.0 - 0.5
        a, fs = d["action"], d["is_first"]
        emb = wm.encoder(o)
        post, prior = wm.rssm.observe(emb, a, fs, None)
        rec = wm.decoder(post["deter"], post["stoch"])
        out["recon"].append(((rec - o) ** 2).sum([2, 3, 4]).mean(1).cpu())
        kl = torch.distributions.kl_divergence(wm.rssm._get_dist(post["logits"]), wm.rssm._get_dist(prior["logits"]))
        out["kl"].append(kl.mean(1).cpu())
        # open loop: observe CTX frames, imagine HOR steps with the logged actions a[CTX:CTX+HOR]
        p_ctx, _ = wm.rssm.observe(emb[:, :CTX], a[:, :CTX], fs[:, :CTX], None)
        init = {k: v[:, -1] for k, v in p_ctx.items()}
        pr = wm.rssm.imagine(a[:, CTX:CTX + HOR], init)
        img = wm.decoder(pr["deter"], pr["stoch"])
        err = ((img - o[:, CTX:CTX + HOR]) ** 2).sum([2, 3, 4])  # (B, HOR)
        out["openloop"].append(err.mean(1).cpu())
        out["openloop_h1_8"].append(err[:, :8].mean(1).cpu())
        out["openloop_h25_32"].append(err[:, 24:].mean(1).cpu())
        if has_heads:
            rhat = agent.two_hot.decode(wm.reward_model(post["deter"], post["stoch"]))
            out["reward_err"].append((rhat - d["reward"]).abs().mean([1, 2]).cpu())
            v_post = agent.two_hot.decode(agent._task_behavior.critic(post["deter"][:, CTX:CTX + HOR], post["stoch"][:, CTX:CTX + HOR]))
            v_pri = agent.two_hot.decode(agent._task_behavior.critic(pr["deter"], pr["stoch"]))
            gap = (v_pri - v_post).abs().squeeze(-1)  # (B, HOR)
            out["value_gap"].append(gap.mean(1).cpu())
            out["value_gap_h25_32"].append(gap[:, 24:].mean(1).cpu())
    res = {}
    for k, v in out.items():
        if v:
            x = torch.cat(v).numpy()
            res[k] = dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), per_chunk=x.tolist())
    return res


def module_groups(sd_prefix_free):
    """Map a flat WM state-dict key (encoder./rssm./decoder.) to a module group name."""
    k = sd_prefix_free
    if k.startswith("encoder."):
        return "encoder"
    if k.startswith("decoder._conv_model"):
        return "decoder.conv"
    if k.startswith("decoder."):
        return "decoder.fc"
    m = re.match(r"rssm\.(_gru\.)?(_\w+)", k)
    if m:
        return "rssm." + ("gru." if m.group(1) else "") + m.group(2)
    return "other"


def flat_wm(sd_wm):
    out = {}
    for part in ["encoder", "rssm", "decoder"]:
        for k, v in sd_wm[part].items():
            out[f"{part}.{k}"] = v.float()
    return out


def drift(pre, ft):
    groups = {}
    for k, v0 in pre.items():
        g = module_groups(k)
        d = groups.setdefault(g, [0.0, 0.0, 0])
        d[0] += float(((ft[k] - v0) ** 2).sum()); d[1] += float((v0 ** 2).sum()); d[2] += v0.numel()
    return {g: dict(rel_l2=(a / b) ** 0.5, params=n) for g, (a, b, n) in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96, help="chunks per data set")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    agent, A = build()
    agent.eval()
    pre_sd = torch.load(PRE, map_location="cpu", weights_only=False)
    snaps = {k: torch.load(f"{RUN}/snapshot_{f}.pt", map_location="cpu", weights_only=False)
             for k, f in [("ft20", 20000), ("ft40", 40000)]}

    online = sorted(glob.glob(f"{RUN}/buffer/*.npz"), key=ep_index)
    sets = {
        "on_bc": [f for f in online if ep_index(f) <= 14],
        "on_rl": [f for f in online if 15 <= ep_index(f) <= 39],
        "on_new": [f for f in online if ep_index(f) >= 40],
        "prior": sorted(glob.glob(f"{RUN}/prior_buffer/*.npz")),
    }
    for emb in ["walker", "cheetah", "cartpole", "acrobot"]:
        sets[emb] = sorted(glob.glob(f"{POOL}/{emb}/*.npz"))[:400]
    data = {}
    for i, (k, files) in enumerate(sets.items()):
        data[k] = load_chunks(files, a.n, np.random.default_rng(100 + i), A)
        print(k, len(files), "episodes ->", data[k]["observation"].shape, flush=True)

    res = {"sets": {k: len(v) for k, v in sets.items()}, "n_chunks": a.n, "error": {}}
    # Part 1: error table
    for which in ["pre", "ft20", "ft40"]:
        load_weights(agent, which, pre_sd, snaps)
        res["error"][which] = {}
        for k, d in data.items():
            torch.manual_seed(0)
            r = evaluate(agent, d, has_heads=(which != "pre" and k.startswith(("on_", "prior"))))
            res["error"][which][k] = r
            print(which, k, {m: round(v["mean"], 3) for m, v in r.items()}, flush=True)

    # Part 2: drift
    P = flat_wm(pre_sd["wm"])
    res["drift"] = {w: drift(P, flat_wm(snaps[w]["wm"])) for w in ["ft20", "ft40"]}
    res["drift"]["ft20_to_ft40"] = drift(flat_wm(snaps["ft20"]["wm"]), flat_wm(snaps["ft40"]["wm"]))
    for w, t in res["drift"].items():
        print("drift", w, {g: round(v["rel_l2"], 4) for g, v in sorted(t.items())}, flush=True)

    # Part 3: module reversion on ft40 (one group back to pretrained weights)
    revert = {
        "encoder": ["encoder."], "decoder": ["decoder."], "rssm.gru": ["rssm._gru."],
        "rssm.prior": ["rssm._prior."], "rssm.posterior": ["rssm._posterior."],
        "encoder+decoder": ["encoder.", "decoder."], "rssm (all)": ["rssm."],
    }
    res["reversion"] = {}
    ft = flat_wm(snaps["ft40"]["wm"])
    for name, prefixes in revert.items():
        load_weights(agent, "ft40", pre_sd, snaps)
        mixed = {k: (P[k] if k.startswith(tuple(prefixes)) else ft[k]) for k in ft}
        for part in ["encoder", "rssm", "decoder"]:
            getattr(agent.wm, part).load_state_dict({k[len(part) + 1:]: v for k, v in mixed.items() if k.startswith(part + ".")})
        res["reversion"][name] = {}
        for k in ["on_new", "prior", "walker"]:
            torch.manual_seed(0)
            r = evaluate(agent, data[k], has_heads=k.startswith(("on_", "prior")))
            res["reversion"][name][k] = {m: dict(mean=v["mean"], se=v["se"]) for m, v in r.items()}
            print("revert", name, k, {m: round(v["mean"], 3) for m, v in r.items()}, flush=True)

    # strip per-chunk lists except for on_new (kept for correlation analysis)
    for w in res["error"]:
        for k in res["error"][w]:
            if k != "on_new":
                for m in res["error"][w][k]:
                    res["error"][w][k][m].pop("per_chunk", None)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
