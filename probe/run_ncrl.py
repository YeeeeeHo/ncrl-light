"""Launch NCRL fine-tuning unchanged except for three things that do not change the learning updates:

1. ckpt_patch.apply(ckpt=True): result-identical, memory-saving imagination (verified bit-identical by equiv_test2.py),
   needed to fit the paper's model (deter 12288 ...) on a 12 GB GPU.
2. Logging without wandb: every evaluation is appended to <work_dir>/eval_log.jsonl with both counters
   (global_step = agent steps, global_frame = steps x action_repeat), the epoch and wall-clock times.
   The time spent in Workspace.__init__ (retrieval + BC pre-training) is logged separately.
3. Full state next to every NCRL snapshot (<snapshot>.full.pt): all agent parameters and buffers, Adam states of the
   world-model / actor / critic optimizers, the actor-critic slow-target counter, the BC agent, Workspace counters and
   the python / numpy / torch / cuda RNG states, and the replay manifest (episode files present at that moment, online
   and prior). (NCRL's own snapshot keeps weights only.) With buffer.save_buffer=true (episodes on disk) this allows a
   controlled branch from the same training state and the same available data. It is NOT a bit-exact resume of the
   whole run: environment RNG and DataLoader worker/prefetch state are not saved.
4. The resolved config after Workspace.__init__ (config_resolved.yaml).

Usage: python -I run_ncrl.py <hydra overrides for src/finetune.py ...>   (run from the directory that should hold logs/)
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch

NCRL = os.environ.get("NCRL_SRC", "/home/yeho/RL/ncrl_src")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(NCRL, "src"))
sys.path.insert(0, HERE)

import ckpt_patch  # noqa: E402
ckpt_patch.apply(ckpt=True)

import finetune  # noqa: E402
from dreamer.dreamer import DreamerAgent  # noqa: E402

_WS = {}
_T0 = time.time()

_orig_init = finetune.Workspace.__init__


def _init(self, cfg):
    t = time.time()
    _orig_init(self, cfg)
    _WS["ws"] = self
    self._probe_init_seconds = time.time() - t
    os.makedirs(self.work_dir, exist_ok=True)
    from omegaconf import OmegaConf
    with open(self.work_dir / "config_resolved.yaml", "w") as f:  # after __init__: sampled seed, adjusted warmup, ...
        f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
    with open(self.work_dir / "run_info.json", "w") as f:
        json.dump(dict(init_seconds_retrieval_and_bc=self._probe_init_seconds, argv=sys.argv,
                       patch="ckpt_patch(ckpt=True)", started=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)


finetune.Workspace.__init__ = _init

_orig_eval = finetune.Workspace.eval


def _scalar(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _eval(self):
    m = _orig_eval(self)
    rec = dict(global_step=self.global_step, global_frame=self.global_frame, epoch=self.global_epoch,
               wall_since_launch=time.time() - _T0, wall_since_train_start=time.time() - self._start_time,
               metrics={k: _scalar(v) for k, v in m.items() if _scalar(v) is not None})
    with open(self.work_dir / "eval_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[probe] eval at step {self.global_step} (frame {self.global_frame}): "
          f"{ {k: v for k, v in rec['metrics'].items() if 'reward' in k or 'success' in k or 'score' in k} }", flush=True)
    return m


finetune.Workspace.eval = _eval

_orig_save = DreamerAgent.save_model


def _save(self, path):
    _orig_save(self, path)
    ws = _WS.get("ws")
    tb = self._task_behavior
    full = dict(agent=self.state_dict(),
                model_opt=self.wm.model_opt._opt.state_dict(),
                actor_opt=tb.actor_opt._opt.state_dict(), critic_opt=tb.critic_opt._opt.state_dict(),
                ac_updates=getattr(tb, "_updates", None),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state_all()))
    if ws is not None:
        full["workspace"] = dict(global_step=ws.global_step, global_episode=ws.global_episode,
                                 global_epoch=ws.global_epoch, global_frame=ws.global_frame,
                                 wall_since_train_start=time.time() - ws._start_time)
        if hasattr(ws, "bc_agent"):
            full["bc_agent"] = ws.bc_agent.state_dict()
        # replay manifest: the episode files that existed at this snapshot, so a later branch reads only these
        # (the run keeps writing new episodes into the same folders)
        man = {}
        for name, d in (("online", ws.work_dir / "buffer"), ("prior", getattr(ws, "offline_buffer_path", None))):
            if d is not None and os.path.isdir(d):
                man[name] = sorted(os.listdir(d))
        full["replay_manifest"] = man
    torch.save(full, str(path).replace(".pt", ".full.pt"))


DreamerAgent.save_model = _save

if __name__ == "__main__":
    # Same as finetune.main() (hydra.main with config_path='../config', config_name='finetune'), but composed with an
    # absolute config dir because hydra cannot locate '../config' when finetune is imported from this launcher.
    # finetune.yaml sets hydra.run.dir='.' and no output subdir, so nothing about the run changes.
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(NCRL, "config"), version_base=None):
        cfg = compose(config_name="finetune", overrides=sys.argv[1:])
    workspace = finetune.Workspace(cfg)
    workspace.train()
