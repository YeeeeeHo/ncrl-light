"""Does the unmasked action padding in NCRL matter?

Code fact (ncrl_src/src/dreamer/nets.py, GaussianPolicy.forward): with use_action_padding the mean is masked
(`mu = mu * action_mask`) but the sample is not -- the masking line has a typo (`sampe = sample * action_mask`).
So in training-mode acting and in imagination, the RSSM receives non-zero values in the padded action dimensions
(DMC Quadruped: dims 12..20; Meta-World: dims 4..20), although every replay/pretraining action has zeros there and
the environment only receives action[:act_dim]. With actor_grad=dynamics the actor gets gradients through those
dims as well.

This probe (forward only, Quadruped Walk 40k snapshot, starts from posterior states of unseen online data):
  1. magnitude of the padded dims of imagined actions;
  2. imagined lambda-return with the original rollout vs the same rollout (same noise) with padded dims zeroed;
  3. norm of the pretrained RSSM input weights for real vs padded action dims.

  python -I padded_action_probe.py --out ../results/padded_action_probe.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_wm_error as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    agent, A = D.build()
    agent.eval()
    pre_sd = torch.load(D.PRE, map_location="cpu", weights_only=False)
    snap = torch.load(f"{D.RUN}/snapshot_40000.pt", map_location="cpu", weights_only=False)
    D.load_weights(agent, "ft40", pre_sd, {"ft40": snap})
    agent._task_behavior.actor.load_state_dict(snap["actor"])
    agent._task_behavior._target_critic.load_state_dict(snap["target_critic"])
    act_dim = 12
    res = {}

    # 3. input weights of the action projection (Linear(21, 1536)): column norms
    for name, sd in [("pretrained", pre_sd["wm"]["rssm"]), ("ft40", snap["wm"]["rssm"])]:
        W = sd["_gru._in_action.0.weight"].float()  # (1536, 21)
        cn = W.norm(dim=0)
        res[f"in_action_colnorm_{name}"] = dict(real=float(cn[:act_dim].mean()), padded=float(cn[act_dim:].mean()),
                                               per_dim=cn.tolist())

    online = [f for f in glob.glob(f"{D.RUN}/buffer/*.npz") if D.ep_index(f) >= 40]
    data = D.load_chunks(online, a.n, np.random.default_rng(7), A)
    wm, ac = agent.wm, agent._task_behavior
    hor = int(ac.cfg.imag_horizon)

    @torch.no_grad()
    def rollout(start, zero_pad, seed):
        torch.manual_seed(seed)
        deter, stoch = start["deter"], start["stoch"]
        seq_d, seq_s, acts = [deter], [stoch], []
        for _ in range(hor):
            pol = ac.actor(deter, stoch)
            act = pol.sample.clone()
            if zero_pad:
                act[..., act_dim:] = 0
            st = wm.rssm.img_step({"deter": deter, "stoch": stoch}, act)
            deter, stoch = st["deter"], st["stoch"]
            seq_d.append(deter); seq_s.append(stoch); acts.append(act)
        sd, ss = torch.stack(seq_d), torch.stack(seq_s)
        rew = agent.two_hot.decode(wm.reward_model(sd, ss))
        val = agent.two_hot.decode(ac._target_critic(sd, ss))
        disc = torch.full_like(rew, ac.cfg.discount)
        import dreamer.dreamer_utils as common
        ret = common.lambda_return(rew[:-1], val[:-1], disc[:-1], bootstrap=val[-1], lambda_=ac.cfg.discount_lambda, axis=0)
        return ret[0].squeeze(-1), torch.stack(acts), rew.squeeze(-1)

    rets, rets0, padmag, realmag = [], [], [], []
    with torch.no_grad():
        for s in range(0, a.n, 16):
            d = {k: torch.as_tensor(v[s:s + 16], device=D.DEV) for k, v in data.items()}
            o = d["observation"].float() / 255.0 - 0.5
            post, _ = wm.rssm.observe(wm.encoder(o), d["action"], d["is_first"], None)
            start = {k: v.reshape((-1,) + v.shape[2:]) for k, v in post.items() if k in ("deter", "stoch")}
            r1, acts, _ = rollout(start, False, s)
            r0, _, _ = rollout(start, True, s)  # same seed -> same policy noise draws for the real dims
            rets.append(r1.cpu()); rets0.append(r0.cpu())
            padmag.append(acts[..., act_dim:].abs().mean().cpu()); realmag.append(acts[..., :act_dim].abs().mean().cpu())
    r1, r0 = torch.cat(rets), torch.cat(rets0)
    res["imagined_return_original"] = dict(mean=float(r1.mean()), se=float(r1.std() / len(r1) ** 0.5))
    res["imagined_return_padded_zeroed"] = dict(mean=float(r0.mean()), se=float(r0.std() / len(r0) ** 0.5))
    diff = r1 - r0
    res["return_diff_original_minus_zeroed"] = dict(mean=float(diff.mean()), se=float(diff.std() / len(diff) ** 0.5),
                                                    frac_positive=float((diff > 0).float().mean()))
    res["abs_action_padded_dims"] = float(torch.stack(padmag).mean())
    res["abs_action_real_dims"] = float(torch.stack(realmag).mean())
    res["n_starts"] = int(len(r1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("in_action")}, indent=1))
    print({k: (v["real"], v["padded"]) for k, v in res.items() if k.startswith("in_action")})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
