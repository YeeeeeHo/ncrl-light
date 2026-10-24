"""Memory / time probe for NCRL's fine-tuning update, with the paper's model size and random weights.

No pretrained checkpoint and no data: the cost of one update does not depend on the weights or pixel values,
so random tensors with the replay buffer's shapes are enough to check (1) whether the original setting fits in
12 GB and (2) where the time of one update goes.

Setting = config/finetune.yaml + the overrides in scripts/finetune.sh:
  rssm.deter 12288, rssm.hidden 1536, rssm.classes 96, encoder.cnn_depth 96, batch 16 x chunk 64,
  imag_horizon 16, actor_grad dynamics, precision 32, action padding to MAX_ACTION_DIM.

  python -I bench_update.py --ncrl /home/yeho/RL/ncrl_src            # paper size
  python -I bench_update.py --ncrl /home/yeho/RL/ncrl_src --small    # default yaml size, for comparison
"""
import argparse
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ncrl", required=True)
    p.add_argument("--small", action="store_true", help="keep finetune.yaml's default model size")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--act_dim", type=int, default=4, help="real action dim (Meta-World 4); padded to MAX_ACTION_DIM")
    p.add_argument("--ckpt", action="store_true", help="apply ckpt_patch (checkpointed imagination, same results)")
    p.add_argument("--ckpt_steps", type=int, default=None, help="with --ckpt: checkpoint only the first n steps")
    p.add_argument("--starts", type=int, default=None, help="timing only: imagine from the first N of the B*T start states")
    p.add_argument("--slim", action="store_true", help="apply ckpt_patch without checkpointing (drop unused imagined tensors only)")
    a = p.parse_args()
    sys.path.insert(0, a.ncrl + "/src")
    from omegaconf import OmegaConf
    import envs
    from dreamer.dreamer import DreamerAgent
    if a.ckpt or a.slim:
        sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
        import ckpt_patch; ckpt_patch.apply(ckpt=a.ckpt, ckpt_steps=a.ckpt_steps)
        print("imagination: " + (f"checkpointed (first {a.ckpt_steps or 'all'} steps) + slim" if a.ckpt else "slim only (no recomputation)"))

    cfg = OmegaConf.load(a.ncrl + "/config/finetune.yaml")
    cfg.ws.mode = "finetune"
    if not a.small:
        cfg.agent.rssm.deter, cfg.agent.rssm.hidden, cfg.agent.rssm.classes = 12288, 1536, 96
        cfg.agent.encoder.cnn_depth = 96
    B, T = cfg.ws.batch_size, cfg.ws.chunk_length
    A = envs.MAX_ACTION_DIM
    dev = "cuda"
    torch.manual_seed(0)
    agent = DreamerAgent(cfg.agent, (3, 64, 64), a.act_dim, A, device=dev)
    n_wm = sum(p.numel() for p in agent.wm.parameters())
    n_ac = sum(p.numel() for p in agent._task_behavior.parameters())
    print(f"model: deter {cfg.agent.rssm.deter}, hidden {cfg.agent.rssm.hidden}, classes {cfg.agent.rssm.classes}, "
          f"cnn {cfg.agent.encoder.cnn_depth} | world model {n_wm/1e6:.1f}M params, actor-critic {n_ac/1e6:.1f}M | batch {B}x{T}",
          flush=True)

    def batch():
        act = np.zeros((B, T, A), np.float32); act[..., :a.act_dim] = np.random.uniform(-1, 1, (B, T, a.act_dim))
        mask = np.zeros((B, T, A), np.float32); mask[..., :a.act_dim] = 1
        first = np.zeros((B, T), bool); first[:, 0] = True
        return dict(observation=np.random.randint(0, 256, (B, T, 3, 64, 64), dtype=np.uint8), action=act, action_mask=mask,
                    reward=np.random.rand(B, T).astype(np.float32), discount=np.ones((B, T), np.float32),
                    is_first=first, is_terminal=np.zeros((B, T), bool))

    ev = lambda: torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for it in range(a.iters + 2):  # first 2 = warm-up (cudnn benchmark, allocator)
        d = batch()
        data = {k: torch.as_tensor(np.copy(v), device=dev) for k, v in d.items()}
        data["observation"] = agent.preprocess_img(data["observation"])
        e = [ev() for _ in range(4)]
        torch.cuda.synchronize(); t0 = time.time()
        e[0].record()
        state, outputs, _ = agent.update_wm(data, 0)          # encoder + RSSM observe + decoder, fwd + bwd + Adam
        e[1].record()
        agent.wm.update_reward(outputs["post"]["deter"].detach(), outputs["post"]["stoch"].detach(), data["reward"])
        e[2].record()
        start = {k: v.detach() for k, v in outputs["post"].items()}
        term = data["is_terminal"]
        if a.starts:  # flatten (B, T) and keep the first N start states, shaped (1, N, ...)
            start = {k: v.reshape((-1,) + tuple(v.shape[2:]))[:a.starts][None] for k, v in start.items()}
            term = term.reshape(-1)[:a.starts][None]
        rf = lambda seq: agent.two_hot.decode(agent.wm.reward_model(seq["deter"], seq["stoch"]))
        agent._task_behavior.update(agent.wm, start, term, rf)  # imagination (B*T or N starts) x 16 + actor/critic
        e[3].record()
        torch.cuda.synchronize()
        if it >= 2:
            rows.append([e[0].elapsed_time(e[1]), e[1].elapsed_time(e[2]), e[2].elapsed_time(e[3]), (time.time() - t0) * 1000])
    r = np.array(rows)
    m = r.mean(0)
    print(f"per update (ms, mean of {len(r)}): world model {m[0]:.0f} | reward head {m[1]:.0f} | actor-critic (imagination) {m[2]:.0f} "
          f"| total {m[3]:.0f}", flush=True)
    print(f"shares: world model {m[0]/m[3]:.0%}, reward {m[1]/m[3]:.0%}, actor-critic {m[2]/m[3]:.0%}")
    print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB)")

    # forward-only split of the world model (no grad), to see encoder / RSSM / decoder shares
    with torch.no_grad():
        fe = [ev() for _ in range(4)]
        fe[0].record(); emb = agent.wm.encoder(data["observation"])
        fe[1].record(); post, prior = agent.wm.rssm.observe(emb, data["action"], data["is_first"], None)
        fe[2].record(); agent.wm.decoder(post["deter"], post["stoch"])
        fe[3].record(); torch.cuda.synchronize()
        print(f"world model forward only (ms): encoder {fe[0].elapsed_time(fe[1]):.0f} | RSSM observe (64 steps) "
              f"{fe[1].elapsed_time(fe[2]):.0f} | decoder {fe[2].elapsed_time(fe[3]):.0f}")
    # acting cost (one env step: encoder + obs_step + actor, batch 1)
    with torch.no_grad():
        lat = agent.wm.rssm.initial(1); act = torch.zeros((1, A), device=dev)
        obs = torch.zeros((1, 3, 64, 64), device=dev); first = torch.ones((1,), device=dev)
        for _ in range(5):
            emb = agent.wm.encoder(obs); lat, _ = agent.wm.rssm.obs_step(lat, act, emb, first, True)
            act = agent._task_behavior.actor(lat["deter"], lat["stoch"]).sample
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50):
            emb = agent.wm.encoder(obs); lat, _ = agent.wm.rssm.obs_step(lat, act, emb, first, True)
            act = agent._task_behavior.actor(lat["deter"], lat["stoch"]).sample
        torch.cuda.synchronize()
        print(f"acting (per env step, batch 1): {(time.time() - t) / 50 * 1000:.1f} ms")


if __name__ == "__main__":
    main()
