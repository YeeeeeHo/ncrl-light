"""Gradient-level equivalence of ckpt_patch vs NCRL's original imagine, with asserts and a raw log.

For each of `--updates` actor-critic updates, two identical agent copies (original imagine / patched imagine) start
from the same start states with the same seed. Recorded per optimizer call (actor_opt, critic_opt): the raw loss
and every parameter gradient right after backward, before clipping/weight decay/step (captured by a copy of NCRL's
Optimizer.__call__ with one recording line added). Then compared: losses, pre-clip gradients, parameters after the
update (actor, critic, slow critic). Fails (exit 1) if any difference is above --tol. Writes a JSON log.

--paper_arch uses the paper's architecture (deter 12288, hidden 1536, classes 96, cnn 96); only the number of
imagination start states is small (--B x --T), so both variants fit in memory. This is a test input, not a change of
the training batch.

  python -I equiv_test2.py --ncrl /home/yeho/RL/ncrl_src --out equiv_small.json
  python -I equiv_test2.py --ncrl /home/yeho/RL/ncrl_src --paper_arch --B 2 --T 8 --out equiv_paper_arch.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

RECORD = []


def recording_call(self, loss, params):
    """Copy of dreamer_utils.Optimizer.__call__ with one added line (RECORD.append ...)."""
    params = list(params)
    assert len(loss.shape) == 0 or (len(loss.shape) == 1 and loss.shape[0] == 1), (self._name, loss.shape)
    metrics = {}
    if self._once:
        count = sum(p.numel() for p in params if p.requires_grad)
        print(f'Found {count} {self._name} parameters.')
        self._once = False
    metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()
    self._scaler.scale(loss).backward()
    self._scaler.unscale_(self._opt)
    RECORD.append((self._name, float(loss.detach()), [None if p.grad is None else p.grad.detach().clone() for p in params]))
    if self._clip:
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        metrics[f'{self._name}_grad_norm'] = norm.item()
    if self._wd:
        self._apply_weight_decay(params)
    self._scaler.step(self._opt)
    self._scaler.update()
    self._opt.zero_grad()
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ncrl", required=True)
    p.add_argument("--paper_arch", action="store_true")
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.ncrl + "/src")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omegaconf import OmegaConf
    import envs
    import dreamer.dreamer_utils as common
    from dreamer.dreamer import DreamerAgent
    import ckpt_patch
    common.Optimizer.__call__ = recording_call

    cfg = OmegaConf.load(a.ncrl + "/config/finetune.yaml")
    cfg.ws.mode = "finetune"
    if a.paper_arch:
        cfg.agent.rssm.deter, cfg.agent.rssm.hidden, cfg.agent.rssm.classes = 12288, 1536, 96
        cfg.agent.encoder.cnn_depth = 96
    B, T, A = a.B, a.T, envs.MAX_ACTION_DIM
    torch.manual_seed(0); np.random.seed(0)
    agent0 = DreamerAgent(cfg.agent, (3, 64, 64), 4, A, device="cuda")
    agent1 = copy.deepcopy(agent0)
    act = np.zeros((B, T, A), np.float32); act[..., :4] = np.random.uniform(-1, 1, (B, T, 4))
    first = np.zeros((B, T), bool); first[:, 0] = True
    obs = agent0.preprocess_img(torch.as_tensor(np.random.randint(0, 256, (B, T, 3, 64, 64), dtype=np.uint8), device="cuda"))
    action, is_first = torch.as_tensor(act, device="cuda"), torch.as_tensor(first, device="cuda")
    is_terminal = torch.zeros((B, T), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        post, _ = agent0.wm.rssm.observe(agent0.wm.encoder(obs), action, is_first, None)
    start = {k: v.detach() for k, v in post.items()}

    def run(agent, patched, seed):
        RECORD.clear()
        if patched:
            ckpt_patch.apply(ckpt=True)
        else:
            ckpt_patch.restore()
        rf = lambda seq: agent.two_hot.decode(agent.wm.reward_model(seq["deter"], seq["stoch"]))
        torch.manual_seed(seed)
        agent._task_behavior.update(agent.wm, start, is_terminal, rf)
        ckpt_patch.restore()
        return list(RECORD)

    log, worst = dict(args=vars(a), start_states=B * T, updates=[]), 0.0
    for u in range(a.updates):
        r0 = run(agent0, False, 100 + u)
        r1 = run(agent1, True, 100 + u)
        assert [x[0] for x in r0] == [x[0] for x in r1], "optimizer call order differs"
        entry = {}
        for (name, l0, g0), (_, l1, g1) in zip(r0, r1):
            gd = max(float((x - y).abs().max()) for x, y in zip(g0, g1) if x is not None)
            gn = max(float(x.abs().max()) for x in g0 if x is not None)
            entry[name] = dict(loss_orig=repr(l0), loss_ckpt=repr(l1), loss_diff=abs(l0 - l1),
                               max_abs_grad_diff=gd, max_abs_grad=gn)
            worst = max(worst, abs(l0 - l1), gd)
        for net in ("actor", "critic", "_target_critic"):
            p0 = dict(getattr(agent0._task_behavior, net).named_parameters())
            p1 = dict(getattr(agent1._task_behavior, net).named_parameters())
            d = max(float((p0[k] - p1[k]).abs().max()) for k in p0)
            entry[f"param_diff_after_{net}"] = d
            worst = max(worst, d)
        log["updates"].append(entry)
        print(f"update {u}: " + json.dumps(entry), flush=True)
    log["max_difference"] = worst
    log["passed"] = worst <= a.tol
    json.dump(log, open(a.out, "w"), indent=1)
    print(f"max difference over everything: {worst:.3e} -> {'PASS' if worst <= a.tol else 'FAIL'} (tol {a.tol})")
    sys.exit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()
