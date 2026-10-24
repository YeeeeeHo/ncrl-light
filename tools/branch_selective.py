"""Experiments A/B pilot: branch NCRL from a saved training state and change ONLY how the world model (WM) is updated.

Runs ncrl_probe/branch_ncrl.py unchanged (restores agent + all Adam states + BC prior + counters + replay manifest,
continues the copy of Workspace.train), after patching two methods:

  --wm_every K   DreamerAgent.update: the WM gradient step runs on updates 0, K, 2K, ... of the branch (K=1 = the
                 original; K=0 = never). On skipped updates the batch is still drawn exactly as before (12 online + 4
                 retrieved chunks), the posterior states are computed with a no-grad forward pass (encoder + RSSM
                 observe) and used for the reward-head update and as the 1024 imagination starts, so the reward head
                 and actor-critic are updated every time, as in NCRL. Env steps per epoch and AC updates per epoch are
                 unchanged; only the number of WM gradient steps changes.
                 NOTE: with K=0 alone, NCRL's reward-head step still applies weight decay (1e-6) to ALL WM params
                 (its optimizer decays every parameter it is given), so the WM shrinks by ~0.25% over 2500 updates.
  --strict_off   K=0 AND the WM is bit-identical to the snapshot: all WM groups are frozen, so the reward-head step
                 decays/steps only the reward head. (Added 2026-10-24 22:3x; the first wm_off arm ran without it.)
  --freeze G     WorldModel.update: the listed module groups get no gradient, no Adam step and no weight decay (also
                 not in the reward-head step, where NCRL's optimizer would otherwise weight-decay all WM params).
                 Groups: encoder, decoder, gru (RSSM recurrent core), prior, posterior.
  --save_end_state NAME   after the last epoch, save a run_ncrl-format full state (<out>/NAME: agent, 3 Adam states,
                 ac counter, BC prior, workspace counters, RNG, replay manifest) and copy config_branch.yaml to
                 config_resolved.yaml, so <out> can be used as --base_run of a later branch.
  --eval_only    no training: evaluate the snapshot's policy exactly like the branch's in-loop evals (fresh eval env
                 with seed cfg.ws.seed+10, --eval_episodes episodes, mean action) -> <out>/eval_start.json.
  --base_run_override / --snapshot_override   replace the values passed after '--' (the queue runner fixes them).

Per epoch (125 updates), mean WM / AC metrics, the number of WM steps and the peak GPU memory of that epoch
(allocated / reserved, reset every epoch) go to <out>/update_metrics.jsonl. Env BRANCH_DEVICE overrides the device
(tests on CPU only).

  python -I branch_selective.py --wm_every 4 -- --base_run <run> --snapshot <run>/snapshot_40000.full.pt --out <dir> --epochs 20
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

NCRL = os.environ.get("NCRL_SRC", "/home/yeho/RL/ncrl_src")
PROBE = "/home/yeho/RL/ncrl_probe"
sys.path.insert(0, os.path.join(NCRL, "src"))
sys.path.insert(0, PROBE)

GROUPS = {"encoder": lambda wm: wm.encoder, "decoder": lambda wm: wm.decoder, "gru": lambda wm: wm.rssm._gru,
          "prior": lambda wm: wm.rssm._prior, "posterior": lambda wm: wm.rssm._posterior}
PREFIX = {"encoder": "wm.encoder.", "decoder": "wm.decoder.", "gru": "wm.rssm._gru.", "prior": "wm.rssm._prior.",
          "posterior": "wm.rssm._posterior."}


def parse(argv):
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_every", type=int, default=1)
    ap.add_argument("--freeze", nargs="*", default=[])
    ap.add_argument("--strict_off", action="store_true")
    ap.add_argument("--save_end_state", default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--snapshot_override", default=None)
    ap.add_argument("--base_run_override", default=None)
    a = ap.parse_args(argv[:split])
    rest = list(argv[split + 1:])
    if a.snapshot_override:
        rest[rest.index("--snapshot") + 1] = a.snapshot_override
    if a.base_run_override:
        rest[rest.index("--base_run") + 1] = a.base_run_override
    if a.strict_off:
        a.wm_every, a.freeze = 0, list(GROUPS)
    return a, rest


def opt_arg(rest, name, default, cast=str):
    return cast(rest[rest.index(name) + 1]) if name in rest else default


def install_patches(a, st):
    """Patch NCRL classes in place (WorldModel.update / update_reward for --freeze, DreamerAgent.update always)."""
    import dreamer.dreamer_utils as common
    from dreamer.dreamer import DreamerAgent, stop_gradient
    from dreamer.wm import WorldModel
    from functools import partial

    if a.freeze:
        def frozen_params(wm):
            return {id(p) for g in a.freeze for p in GROUPS[g](wm).parameters()}

        def wm_update(self, data, state=None):
            fz = frozen_params(self)
            with common.RequiresGrad(self):
                for g in a.freeze:
                    GROUPS[g](self).requires_grad_(False)
                with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
                    model_loss, state, outputs, metrics = self.loss(data, state)
                metrics.update(self.model_opt(model_loss, [p for p in self.parameters() if id(p) not in fz]))
            return state, outputs, metrics
        WorldModel.update = wm_update

        def update_reward(self, deter, stoch, rew_target):
            fz = frozen_params(self)
            with common.RequiresGrad(self):
                with torch.amp.autocast(device_type='cuda', enabled=self._use_amp):
                    logits = self.reward_model(deter, stoch)
                    loss = self.two_hot.loss(logits, rew_target)
                metrics = self.model_opt(loss, [p for p in self.parameters() if id(p) not in fz])
            return metrics
        WorldModel.update_reward = update_reward

    def agent_update(self, online_data, offline_data, step):
        # identical to DreamerAgent.update except for the gated WM step
        if (not online_data) and offline_data:
            data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in offline_data.items()}
        elif online_data and (not offline_data):
            online_data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in online_data.items()}
            data = online_data
        else:
            data = {k: torch.cat([v, offline_data[k]], dim=0) for k, v in online_data.items()}
            data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in data.items()}
            online_data = {k: torch.as_tensor(np.copy(v), device=self.device) for k, v in online_data.items()}
        data['observation'] = self.preprocess_img(data['observation'])

        do_wm = a.wm_every > 0 and st["n"] % a.wm_every == 0
        if do_wm:
            state, outputs, metrics = self.update_wm(data, step)
            st["wm_steps"] += 1
        else:
            with torch.no_grad():
                embed = self.wm.encoder(data['observation'])
                post, _ = self.wm.rssm.observe(embed, data['action'], data['is_first'], None)
            state, outputs, metrics = None, {'post': post, 'is_terminal': data['is_terminal']}, {}
        st["n"] += 1
        st["agent"] = self

        num_online = online_data['reward'].shape[0]
        metrics.update(self.wm.update_reward(outputs['post']['deter'][:num_online].detach(),
                                             outputs['post']['stoch'][:num_online].detach(), online_data['reward']))
        start = {k: stop_gradient(v) for k, v in outputs['post'].items()}

        def reward_fn(seq):
            return self.two_hot.decode(self.wm.reward_model(seq['deter'], seq['stoch']))
        metrics.update(self._task_behavior.update(self.wm, start, data['is_terminal'], partial(reward_fn)))

        for k, v in metrics.items():
            try:
                st["acc"].setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass
        if st["n"] % st.get("per_epoch", 125) == 0:  # one epoch of Quadruped Walk updates
            rec = dict(updates=st["n"], wm_steps=st["wm_steps"])
            if torch.cuda.is_available():
                rec["peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
                rec["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
                torch.cuda.reset_peak_memory_stats()
            rec.update({k: float(np.mean(v)) for k, v in st["acc"].items()})
            if st.get("log") is not None:
                st["log"].write(json.dumps(rec) + "\n"); st["log"].flush()
            st["acc"] = {}
        return state, metrics
    DreamerAgent.update = agent_update


def capture_workspace(st):
    import finetune
    orig = finetune.Workspace.eval

    def _eval(self):
        st["ws"] = self
        return orig(self)
    finetune.Workspace.eval = _eval


def save_full_state(ws, path):
    """Same content/keys as ncrl_probe/run_ncrl.py `_save` (<snapshot>.full.pt), readable by branch_ncrl.py."""
    agent = ws.agent
    tb = agent._task_behavior
    full = dict(agent=agent.state_dict(), model_opt=agent.wm.model_opt._opt.state_dict(),
                actor_opt=tb.actor_opt._opt.state_dict(), critic_opt=tb.critic_opt._opt.state_dict(),
                ac_updates=getattr(tb, "_updates", None),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    full["workspace"] = dict(global_step=ws.global_step, global_episode=ws.global_episode,
                             global_epoch=ws.global_epoch, global_frame=ws.global_frame,
                             wall_since_train_start=time.time() - ws._start_time)
    if hasattr(ws, "bc_agent"):
        full["bc_agent"] = ws.bc_agent.state_dict()
    man = {}
    for name, d in (("online", Path(ws.work_dir) / "buffer"), ("prior", getattr(ws, "offline_buffer_path", None))):
        if d is not None and os.path.isdir(d):
            man[name] = sorted(os.listdir(d))
    full["replay_manifest"] = man
    torch.save(full, path)
    return full


class _RecordingEnv:
    """Pass-through env wrapper that records the return of every episode (for per-episode eval returns)."""
    def __init__(self, env):
        self._env, self.returns = env, []

    def reset(self):
        self.returns.append(0.0)
        return self._env.reset()

    def step(self, action):
        ts = self._env.step(action)
        self.returns[-1] += float(ts.reward)
        return ts

    def __getattr__(self, k):
        return getattr(self._env, k)


def build_ws_for_eval(base_run, snap_path, out, seed, eval_episodes):
    """Minimal Workspace with the snapshot's agent and the branch's eval env (seed cfg.ws.seed + 10)."""
    from omegaconf import OmegaConf, open_dict
    import envs
    import finetune
    from dreamer.dreamer import DreamerAgent
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = OmegaConf.load(os.path.join(base_run, "config_resolved.yaml"))
    with open_dict(cfg):
        cfg.ws.num_eval_episodes = eval_episodes
        cfg.ws.save_video = False
        if os.environ.get("BRANCH_DEVICE"):
            cfg.ws.device = os.environ["BRANCH_DEVICE"]
    full = torch.load(snap_path, map_location="cpu", weights_only=False)
    ws = finetune.Workspace.__new__(finetune.Workspace)
    ws.cfg, ws.work_dir, ws.device = cfg, Path(out), torch.device(cfg.ws.device)
    ws.eval_env = _RecordingEnv(envs.make_env(cfg.env, cfg.ws.seed + 10))
    ws.obs_shape = ws.eval_env.obs_space.image.shape
    ws.act_dim = ws.eval_env.action_space.shape[0]
    padded = envs.MAX_ACTION_DIM if cfg.ws.use_action_padding else ws.act_dim
    ws.agent = DreamerAgent(cfg.agent, ws.obs_shape, ws.act_dim, padded, device=cfg.ws.device)
    ws.agent.load_state_dict({k: v.to(ws.device) for k, v in full["agent"].items()})
    w = full["workspace"]
    ws._global_step, ws._global_episode, ws._global_epoch = w["global_step"], w["global_episode"], w["global_epoch"]
    ws._start_time = time.time()
    return ws, full


def eval_only(a, rest):
    base_run, snap = opt_arg(rest, "--base_run", None), opt_arg(rest, "--snapshot", None)
    out = opt_arg(rest, "--out", None)
    seed, n = opt_arg(rest, "--seed", 7, int), opt_arg(rest, "--eval_episodes", 10, int)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    ws, full = build_ws_for_eval(base_run, snap, out, seed, n)
    m = ws.eval()
    rets = ws.eval_env.returns
    rec = dict(snapshot=snap, base_run=base_run, global_frame=ws.global_frame, global_step=ws.global_step,
               epoch=ws.global_epoch, eval_episodes=n, seed=seed, eval_env_seed=int(ws.cfg.ws.seed) + 10,
               avg_episode_reward=float(m["eval/avg_episode_reward"]), episode_returns=rets,
               se=float(np.std(rets, ddof=1) / np.sqrt(len(rets))) if len(rets) > 1 else None,
               device=str(ws.device), seconds=time.time() - t0)
    with open(os.path.join(out, "eval_start.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("eval_start", {k: v for k, v in rec.items() if k != "episode_returns"}, flush=True)


def main():
    a, rest = parse(sys.argv[1:])
    if a.eval_only:
        eval_only(a, rest)
        return
    out_dir = opt_arg(rest, "--out", None)
    os.makedirs(out_dir, exist_ok=True)
    st = {"n": 0, "wm_steps": 0, "acc": {}, "log": open(os.path.join(out_dir, "update_metrics.jsonl"), "a")}
    with open(os.path.join(out_dir, "selective_args.json"), "w") as f:
        json.dump(dict(wm_every=a.wm_every, freeze=a.freeze, strict_off=a.strict_off,
                       save_end_state=a.save_end_state, branch_args=rest), f)
    install_patches(a, st)
    capture_workspace(st)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    import branch_ncrl
    sys.argv = ["branch_ncrl.py"] + rest
    branch_ncrl.main()

    if a.save_end_state:
        ws = st["ws"]
        p = os.path.join(out_dir, a.save_end_state)
        save_full_state(ws, p)
        shutil.copy2(os.path.join(out_dir, "config_branch.yaml"), os.path.join(out_dir, "config_resolved.yaml"))
        print("saved end state", p, "epoch", ws.global_epoch, "frame", ws.global_frame, flush=True)

    # end-of-branch check: relative change of each WM group vs the snapshot (frozen groups must be exactly 0)
    snap = torch.load(opt_arg(rest, "--snapshot", None), map_location="cpu", weights_only=False)["agent"]
    wm = st["agent"].wm
    end = {"wm_steps": st["wm_steps"], "updates": st["n"], "strict_off": a.strict_off}
    for g, get in GROUPS.items():
        num = den = 0.0
        maxabs = 0.0
        for k, v in get(wm).state_dict().items():
            v0 = snap[PREFIX[g] + k].float()
            dv = v.detach().cpu().float() - v0
            num += float((dv ** 2).sum()); den += float((v0 ** 2).sum()); maxabs = max(maxabs, float(dv.abs().max()))
        end[f"rel_change_{g}"] = (num / den) ** 0.5
        end[f"max_abs_change_{g}"] = maxabs
    with open(os.path.join(out_dir, "selective_end.json"), "w") as f:
        json.dump(end, f, indent=1)
    print("selective_end", end, flush=True)


if __name__ == "__main__":
    main()
