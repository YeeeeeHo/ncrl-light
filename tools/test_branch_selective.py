"""CPU tests for branch_selective.py (no GPU use: the GPU is busy with the pilot).

A  (--mode unit): small NCRL model (finetune.yaml default size) on CPU, 3 updates with random batches:
     strict_off -> every WM tensor bit-identical; reward head / actor / critic changed
     wm_every 0 (old wm_off) -> WM shrinks by exactly (1-1e-6)^3 (NCRL reward-step weight decay), no other change
     wm_every 2 -> WM steps on updates 0 and 2 only
B  (--mode eval): eval_only on a real snapshot (paper size, CPU, OSMesa rendering), 1 episode
C  (--mode resume): save_full_state from a CPU workspace built from the 40k snapshot, then branch_ncrl.main with
     --epochs 0 using that folder as --base_run (checks keys/format and the replay-manifest copy)

  MUJOCO_GL=osmesa python -I test_branch_selective.py --mode unit
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import branch_selective as BS  # noqa: E402

RUN = "/home/yeho/RL/ncrl_runs/baseline/logs/finetune/dmc-quadruped-walk/09af53c751cc43f5b4f6d07d6a082f14_1"


def unit_case(case):
    from omegaconf import OmegaConf
    import envs
    from dreamer.dreamer import DreamerAgent
    argv = {"strict": ["--strict_off", "--"], "off": ["--wm_every", "0", "--"], "every2": ["--wm_every", "2", "--"]}[case]
    a, _ = BS.parse(argv + ["--out", "x"])
    st = {"n": 0, "wm_steps": 0, "acc": {}, "log": None, "per_epoch": 10**9}
    BS.install_patches(a, st)
    cfg = OmegaConf.load(os.path.join(BS.NCRL, "config/finetune.yaml"))
    cfg.ws.mode = "finetune"
    torch.manual_seed(0); np.random.seed(0)
    agent = DreamerAgent(cfg.agent, (3, 64, 64), 12, envs.MAX_ACTION_DIM, device="cpu")
    snap = {k: v.clone() for k, v in agent.state_dict().items()}
    B, T, A = 4, 8, envs.MAX_ACTION_DIM

    def batch(n):
        act = np.zeros((n, T, A), np.float32); act[..., :12] = np.random.uniform(-1, 1, (n, T, 12))
        first = np.zeros((n, T), bool); first[:, 0] = True
        d = dict(observation=np.random.randint(0, 256, (n, T, 3, 64, 64), dtype=np.uint8), action=act,
                 action_mask=(act != 0).astype(np.float32), reward=np.random.rand(n, T, 1).astype(np.float32),
                 discount=np.ones((n, T, 1), np.float32), is_first=first, is_terminal=np.zeros((n, T, 1), bool))
        return {k: torch.as_tensor(v) for k, v in d.items()}  # the replay DataLoader yields torch tensors
    wm_counts = []
    for _ in range(3):
        before = st["wm_steps"]
        agent.update(batch(3), batch(1), 0)
        wm_counts.append(st["wm_steps"] - before)
    now = agent.state_dict()
    res = {"case": case, "wm_steps_per_update": wm_counts}
    for group in ["wm.encoder.", "wm.rssm.", "wm.decoder.", "wm.reward_model.", "_task_behavior.actor.",
                  "_task_behavior.critic."]:
        keys = [k for k in now if k.startswith(group) and now[k].dtype.is_floating_point]
        identical = all(torch.equal(now[k], snap[k]) for k in keys)
        ratio = [float((now[k].double() / snap[k].double())[snap[k] != 0].mean()) for k in keys[:3]]
        res[group] = dict(identical=identical, mean_ratio_first3=ratio)
    return res


def run_unit():
    out = {}
    for case in ["strict", "off", "every2"]:
        r = subprocess.run([sys.executable, "-I", __file__, "--mode", "unit_case", "--case", case],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        out[case] = json.loads(line[-1]) if line else {"error": r.stderr[-2000:]}
    shrink = float(np.float32(1 - 1e-6)) ** 3  # NCRL multiplies fp32 params by fp32(1-1e-6) each step
    checks = {
        "strict: encoder/rssm/decoder identical": all(out["strict"][g]["identical"] for g in ["wm.encoder.", "wm.rssm.", "wm.decoder."]),
        "strict: reward/actor/critic changed": not any(out["strict"][g]["identical"] for g in ["wm.reward_model.", "_task_behavior.actor.", "_task_behavior.critic."]),
        "strict: 0 WM steps": out["strict"]["wm_steps_per_update"] == [0, 0, 0],
        "off: WM not identical (decay only)": not out["off"]["wm.rssm."]["identical"],
        "off: WM ratio == (1-1e-6)^3": all(abs(x - shrink) < 1e-7 for g in ["wm.encoder.", "wm.rssm.", "wm.decoder."] for x in out["off"][g]["mean_ratio_first3"]),
        "every2: WM steps [1,0,1]": out["every2"]["wm_steps_per_update"] == [1, 0, 1],
    }
    print(json.dumps(out, indent=1))
    print(json.dumps(checks, indent=1))
    return all(checks.values()), {"cases": out, "checks": checks}


def run_eval(tmp):
    os.environ["BRANCH_DEVICE"] = "cpu"
    sys.argv = ["branch_selective.py", "--eval_only", "--", "--base_run", RUN, "--snapshot", f"{RUN}/snapshot_40000.full.pt",
                "--out", os.path.join(tmp, "eval40"), "--eval_episodes", "1"]
    BS.main()
    rec = json.load(open(os.path.join(tmp, "eval40", "eval_start.json")))
    ok = rec["eval_episodes"] == 1 and len(rec["episode_returns"]) == 1 and rec["global_frame"] == 40000
    return ok, rec


def run_resume(tmp):
    os.environ["BRANCH_DEVICE"] = "cpu"
    src = os.path.join(tmp, "fake_base")
    os.makedirs(src, exist_ok=True)
    for d in ["buffer", "prior_buffer"]:
        if not os.path.exists(os.path.join(src, d)):
            os.symlink(os.path.join(RUN, d), os.path.join(src, d))
    ws, _ = BS.build_ws_for_eval(RUN, f"{RUN}/snapshot_40000.full.pt", src, 7, 1)
    ws.offline_buffer_path = os.path.join(src, "prior_buffer")
    full = BS.save_full_state(ws, os.path.join(src, "state_test.full.pt"))
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(os.path.join(RUN, "config_resolved.yaml"))
    with open_dict(cfg):
        cfg.ws.device = "cpu"
    OmegaConf.save(cfg, os.path.join(src, "config_resolved.yaml"))
    import branch_ncrl
    out = os.path.join(tmp, "resume_out")
    sys.argv = ["branch_ncrl.py", "--base_run", src, "--snapshot", os.path.join(src, "state_test.full.pt"), "--out", out,
                "--epochs", "0"]
    branch_ncrl.main()
    rows = [json.loads(l) for l in open(os.path.join(out, "branch_log.jsonl"))]
    n_online = len(os.listdir(os.path.join(out, "buffer"))); n_prior = len(os.listdir(os.path.join(out, "prior_buffer")))
    ok = rows[0]["kind"] == "start" and rows[-1]["kind"] == "end" and rows[0]["epoch"] == 40 and \
        n_online == len(full["replay_manifest"]["online"]) and n_prior == len(full["replay_manifest"]["prior"])
    return ok, dict(keys=sorted(full.keys()), start=rows[0], n_online=n_online, n_prior=n_prior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--tmp", default="/tmp/claude-1000/-home-yeho--claude-spares-spare-mDbVz8/be2f29b6-c151-4527-a58b-288dc9c3032e/scratchpad/bs_test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "unit_case":
        print(json.dumps(unit_case(a.case)))
        return
    os.makedirs(a.tmp, exist_ok=True)
    ok, info = {"unit": run_unit, "eval": lambda: run_eval(a.tmp), "resume": lambda: run_resume(a.tmp)}[a.mode]()
    print("PASS" if ok else "FAIL", a.mode)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(mode=a.mode, passed=ok, info=info), f, indent=1, default=str)


if __name__ == "__main__":
    main()
