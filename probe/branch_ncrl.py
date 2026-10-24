"""Branch NCRL fine-tuning from a saved training state (<snapshot>.full.pt from run_ncrl.py) and continue it with
either the original imagination (all B*T = 1024 start states) or a subset of N start states.

Restored from the snapshot: all agent parameters/buffers, the three Adam states (world model, actor, critic), the
actor-critic slow-target counter, the BC policy (no BC re-training), the Workspace counters (step, episode, epoch),
and the replay data that existed at the snapshot (the manifest's online and prior episode files are COPIED into the
branch folder; the base run's folders are only read). NOT restored: env RNG, DataLoader state; the branch seed
(--seed) sets python/numpy/torch RNG, and every arm uses the same --seed, so arms differ only in the start-state rule.

The loop below is a copy of finetune.Workspace.train with three changes: it starts at the saved epoch + 1, stops
after --epochs epochs, and (for --starts N) ActorCritic.update receives N start states drawn uniformly without
replacement from the B*T posterior states (world-model update unchanged). Arms with --starts use ckpt_patch without
recomputation (fits in 12 GB); the full arm needs ckpt_patch with checkpointing. Both are result-identical to NCRL.

  python -I branch_ncrl.py --base_run <work_dir> --snapshot <work_dir>/snapshot_40000.full.pt --out <dir> --epochs 40 [--starts 256]
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch

NCRL = os.environ.get("NCRL_SRC", "/home/yeho/RL/ncrl_src")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(NCRL, "src"))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--starts", type=int, default=None, help="imagination start states per update (None = all B*T)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval_episodes", type=int, default=10)
    a = p.parse_args()

    import ckpt_patch
    ckpt_patch.apply(ckpt=a.starts is None)  # full arm needs recomputation to fit; subset arms do not
    from omegaconf import OmegaConf, open_dict
    import envs
    import utils
    import finetune
    from dreamer.dreamer import DreamerAgent
    from dreamer.actor_critic import ActorCritic
    from behavior_cloning import BCAgent

    if a.starts is not None:
        orig_update = ActorCritic.update

        def subset_update(self, world_model, start, is_terminal, reward_fn):
            flat = {k: v.reshape((-1,) + tuple(v.shape[2:])) for k, v in start.items()}
            n = flat["deter"].shape[0]
            idx = torch.randperm(n, device=flat["deter"].device)[:a.starts]
            sub = {k: v[idx][None] for k, v in flat.items()}
            return orig_update(self, world_model, sub, is_terminal.reshape(-1)[idx][None], reward_fn)
        ActorCritic.update = subset_update

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = OmegaConf.load(os.path.join(a.base_run, "config_resolved.yaml"))
    full = torch.load(a.snapshot, map_location="cpu", weights_only=False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.ws.num_eval_episodes = a.eval_episodes
    OmegaConf.save(cfg, out / "config_branch.yaml")

    ws = finetune.Workspace.__new__(finetune.Workspace)
    ws.cfg, ws.work_dir, ws.device = cfg, out, torch.device(cfg.ws.device)
    ws.train_env = envs.make_env(cfg.env, cfg.ws.seed + a.seed)
    ws.eval_env = envs.make_env(cfg.env, cfg.ws.seed + 10)          # same eval env seed as the base run
    obs_shape = ws.obs_shape = ws.train_env.obs_space.image.shape
    ws.act_dim = ws.train_env.action_space.shape[0]
    padded = envs.MAX_ACTION_DIM if cfg.ws.use_action_padding else ws.act_dim
    ws.agent = DreamerAgent(cfg.agent, obs_shape, ws.act_dim, padded, device=cfg.ws.device)
    ws.agent.load_state_dict({k: v.to(ws.device) for k, v in full["agent"].items()})
    tb = ws.agent._task_behavior
    ws.agent.wm.model_opt._opt.load_state_dict(full["model_opt"])
    tb.actor_opt._opt.load_state_dict(full["actor_opt"])
    tb.critic_opt._opt.load_state_dict(full["critic_opt"])
    tb._updates = full["ac_updates"]
    del full["agent"], full["model_opt"], full["actor_opt"], full["critic_opt"]

    # replay data at the snapshot: copy manifest files into the branch folders
    man = full["replay_manifest"]
    prior_src = Path(cfg.buffer.offline_path) if cfg.buffer.offline_path else Path(a.base_run) / "prior_buffer"
    for name, src, dst in (("online", Path(a.base_run) / "buffer", out / "buffer"), ("prior", prior_src, out / "prior_buffer")):
        dst.mkdir(exist_ok=True)
        for f in man[name]:
            if not (dst / f).exists():
                shutil.copy2(src / f, dst / f)
    ws.replay_storage, ws.replay_loader = ws._setup_buffer(cfg.buffer, out / "buffer", obs_shape, padded,
                                                           int(cfg.buffer.online_ratio * cfg.buffer.batch_size), True)
    ws.offline_buffer_path = out / "prior_buffer"
    ws.offline_storage, ws.offline_loader = ws._setup_buffer(cfg.buffer, out / "prior_buffer", obs_shape, padded,
                                                             int((1. - cfg.buffer.online_ratio) * cfg.buffer.batch_size), True)
    ws._replay_iter = ws._offline_iter = ws._bc_iter = None
    if "bc_agent" in full:
        ws.bc_agent = BCAgent((9, 64, 64), padded).to(ws.device)
        ws.bc_agent.load_state_dict(full["bc_agent"])
    w = full["workspace"]
    ws._global_step, ws._global_episode, ws._global_epoch = w["global_step"], w["global_episode"], w["global_epoch"]
    ws._start_time = time.time()
    t_launch = time.time()

    log = open(out / "branch_log.jsonl", "a")

    def write(kind, **kw):
        log.write(json.dumps(dict(kind=kind, global_step=ws.global_step, global_frame=ws.global_frame,
                                  epoch=ws.global_epoch, wall=time.time() - t_launch, **kw)) + "\n"); log.flush()
    write("start", starts=a.starts, seed=a.seed, snapshot=a.snapshot)

    # ---- copy of Workspace.train, starting at the saved epoch ----
    self = ws
    FRAMES_PER_EPOCH = self.train_env.episode_length
    STEPS_PER_EPOCH = ceil(FRAMES_PER_EPOCH / self.cfg.env.action_repeat)
    WARMUP_EPOCH = ceil(self.cfg.ws.warmup_frames / FRAMES_PER_EPOCH)
    GRAD_PER_EPOCH = ceil(STEPS_PER_EPOCH / (self.cfg.ws.batch_size * self.cfg.ws.chunk_length / self.cfg.ws.train_ratio))
    EVAL_EVERY_EPOCH = ceil(self.cfg.ws.eval_every_frames / FRAMES_PER_EPOCH)
    RECON_EVERY_EPOCH = ceil(self.cfg.ws.recon_every_frames / FRAMES_PER_EPOCH)
    first = self.global_epoch + 1
    for epoch in range(first, first + a.epochs):
        self._global_epoch = epoch
        t_ep = time.time()
        episode_step, episode_reward = 0, 0
        timestep = self.train_env.reset()
        agent_state = None
        self.replay_storage.add(timestep)
        if hasattr(self, 'bc_agent'):
            FRAME_STACK = 3
            from collections import deque
            bc_obs_buffer = deque([timestep.observation] * FRAME_STACK, maxlen=FRAME_STACK)
            if random.random() < utils.schedule(self.cfg.ws.jsrl_schedule, self.global_step):
                _env_step = self.train_env.episode_length // self.cfg.env.action_repeat
                _warmup_steps = [(_env_step / 10) * i for i in range(1, 11)]
                BC_WARMUP_STEPS = random.choice(_warmup_steps)
                BC_START_STEP = random.randint(0, max(0, _env_step - BC_WARMUP_STEPS))
            else:
                BC_WARMUP_STEPS, BC_START_STEP = 0, 0
        while not bool(timestep.done):
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.global_epoch > WARMUP_EPOCH:
                    action, latent = self.agent.act(timestep, self.global_step, eval_mode=False, state=agent_state)
                    if hasattr(self, 'bc_agent') and BC_START_STEP < episode_step < BC_WARMUP_STEPS + BC_START_STEP:
                        action = self.bc_agent.act(torch.from_numpy(np.concatenate(list(bc_obs_buffer))).to(self.device))
                    agent_state = (latent, action)
                    action = action.cpu().numpy()[0][:self.act_dim]
                else:
                    if hasattr(self, 'bc_agent'):
                        action, agent_state = self.bc_agent.act(torch.from_numpy(
                            np.concatenate(list(bc_obs_buffer))).to(self.device))[0].cpu().numpy()[:self.act_dim], None
                    else:
                        action = np.random.uniform(-1, 1, size=(self.act_dim,)).astype(np.float32)
            timestep = self.train_env.step(action)
            self.replay_storage.add(timestep)
            if hasattr(self, 'bc_agent'):
                bc_obs_buffer.append(timestep.observation)
            episode_reward += timestep.reward
            episode_step += 1
            self._global_step += 1
        self._global_episode += 1
        t_collect = time.time() - t_ep
        t_up = time.time()
        if self.global_epoch > min(WARMUP_EPOCH, 10):
            for _ in range(GRAD_PER_EPOCH):
                self.agent.update(online_data=None if self.cfg.buffer.online_ratio == 0. else next(self.replay_iter),
                                  offline_data=None if self.cfg.buffer.online_ratio == 1. else next(self.offline_iter),
                                  step=self.global_step)
        torch.cuda.synchronize()
        t_update = time.time() - t_up
        if self.global_epoch > min(WARMUP_EPOCH, 10) and self.global_epoch % RECON_EVERY_EPOCH == 0:
            self.agent.report(next(self.replay_iter))   # kept: the original loop does this compute too
        write("epoch", train_return=float(episode_reward), collect_s=t_collect, update_s=t_update,
              updates=GRAD_PER_EPOCH, epoch_s=time.time() - t_ep)
        if self.global_epoch % EVAL_EVERY_EPOCH == 0:
            m = self.eval()
            write("eval", **{k: float(v) for k, v in m.items() if k != 'eval/video' and np.isscalar(v)})
            print(f"epoch {epoch}: eval {m.get(f'eval/{cfg.env.task}/episode_reward')}", flush=True)
    write("end")


if __name__ == "__main__":
    main()
