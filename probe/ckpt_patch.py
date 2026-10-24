"""Memory-saving, result-preserving imagination for NCRL (does not edit the NCRL repo; patches at import time).

NCRL's WorldModel.imagine unrolls `horizon` steps of (policy sample -> RSSM img_step) on all B*T start states and
keeps every intermediate activation for the backward pass (actor_grad = dynamics). At the paper size that runs out of
memory on a 12 GB GPU. This patch wraps each step in torch.utils.checkpoint with the RNG state preserved, so the
backward recomputes the step's activations instead of storing them. The forward values, the random draws and the
gradients are the same as the original; only memory and time change. Everything else (optimizer steps, gradient
clipping, slow-target updates, critic update) is untouched because only `imagine` is replaced.

    import ckpt_patch; ckpt_patch.apply()      # after NCRL's src/ is on sys.path
"""
import torch
from torch.utils.checkpoint import checkpoint


def _imagine_ckpt(self, policy, start, is_terminal, horizon, eval_policy=False):
    """Same computation as dreamer/wm.py WorldModel.imagine, with two memory changes that leave every value used
    downstream unchanged:
      - each step (policy sample -> img_step) is checkpointed with the RNG state preserved;
      - the imagined 'feat' and 'logits' are not kept: ActorCritic.update / target / losses never read them
        (only deter, stoch, action, reward, discount, weight; 'feat' was used only for its shape).
    """
    stop_gradient = lambda x: x.detach()
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items() if v is not None}
    task_id = start['task_id'] if 'task_id' in start.keys() else None
    assert task_id is None, "task_id path not covered by the equivalence test"

    action0 = torch.zeros_like(policy(start['deter'], start['stoch'], task_id).mean, device=self.device)
    seq = {'deter': [start['deter']], 'stoch': [start['stoch']], 'action': [action0]}

    def step(deter, stoch):
        policy_output = policy(stop_gradient(deter), stop_gradient(stoch), task_id)
        action = policy_output.sample if not eval_policy else policy_output.mean
        state = self.rssm.img_step({'stoch': stoch, 'deter': deter}, action)  # img_step reads only these two
        return state['stoch'], state['deter'], action

    n_ckpt = horizon if _MODE.get("ckpt_steps") is None else _MODE["ckpt_steps"]
    for i in range(horizon):
        if _MODE["ckpt"] and i < n_ckpt:
            stoch, deter, action = checkpoint(step, seq['deter'][-1], seq['stoch'][-1],
                                              use_reentrant=False, preserve_rng_state=True)
        else:
            stoch, deter, action = step(seq['deter'][-1], seq['stoch'][-1])
        seq['stoch'].append(stoch); seq['deter'].append(deter); seq['action'].append(action)

    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    disc = self.cfg.discount * torch.ones(list(seq['deter'].shape[:-1]) + [1], device=self.device)
    seq['discount'] = disc
    seq['weight'] = torch.cumprod(torch.cat([torch.ones_like(disc[:1], device=self.device), disc[:-1]], 0), 0)
    return seq


_MODE = {"ckpt": True, "ckpt_steps": None}


def apply(ckpt=True, ckpt_steps=None):
    """ckpt=True: checkpointed steps + slim storage (ckpt_steps: checkpoint only the first n imagination steps,
    None = all). ckpt=False: slim storage only (no recomputation). Results are identical in every mode."""
    _MODE["ckpt"], _MODE["ckpt_steps"] = ckpt, ckpt_steps
    import dreamer.wm as wm
    wm.WorldModel._imagine_original = wm.WorldModel.imagine
    wm.WorldModel.imagine = _imagine_ckpt


def restore():
    import dreamer.wm as wm
    if hasattr(wm.WorldModel, '_imagine_original'):
        wm.WorldModel.imagine = wm.WorldModel._imagine_original
