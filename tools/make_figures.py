"""README figures, generated from the result files (re-run after new results):
  docs/figures/update_time.png   - where one NCRL update spends time (results/profile_update_run1.json)
  docs/figures/wm_error.png      - world-model open-loop error on seen vs unseen online data (results/diag_wm_error.json)
  docs/figures/pilot_tradeoff.png - branch pilot: update time vs eval return, per branch point (runs/branch40k)

Colors: the dataviz reference palette's first three categorical slots (validated all-pairs there; the validator
script itself needs node, which this machine does not have). Labels are in English to avoid missing CJK fonts.

  python -I make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "docs", "figures")
sys.path.insert(0, HERE)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def update_time():
    r = json.load(open(os.path.join(ROOT, "results", "profile_update_run1.json")))
    wm, ac = r["wm_update_ms"], r["ac_update_ms"]["ckpt_1024"]
    rows = [
        ("WM: backward (all 285.7M params)", wm["wm_bwd"], C1),
        ("WM: RSSM 64-step scan, forward", wm["wm_fwd_rssm_observe"], C1),
        ("WM: decoder + encoder forward", wm["wm_fwd_decoder"] + wm["wm_fwd_encoder"] + wm["wm_fwd_kl"] + wm["wm_fwd_recon_loss"], C1),
        ("WM: Adam + clip + weight decay", wm["wm_opt_adam"] + wm["wm_opt_clip"] + wm["wm_opt_weight_decay"], C1),
        ("AC: actor backward through RSSM*", ac["actor_bwd_through_dynamics"], C2),
        ("AC: imagination rollout, 1024 starts x 16", ac["imag_rollout_fwd"], C2),
        ("AC: reward / value / lambda-return / actor loss", ac["imag_reward_value_target_fwd"] + ac["actor_loss_fwd"], C2),
        ("AC: critic + optimizers + reward head", ac["critic_fwd"] + ac["critic_bwd"] + ac["critic_opt"] + ac["actor_opt"]
         + ac["slow_target"] + ac["reward_head_update"], C2),
    ]
    total = sum(v for _, v, _ in rows)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [v for _, v, _ in rows], height=0.55, color=[c for *_, c in rows])
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v + 12, yi, f"{v:.0f} ms", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [n for n, _, _ in rows], color=INK)
    ax.set_xlabel("milliseconds per update (median of 8, CUDA events)")
    wm_sum = sum(v for _, v, c in rows if c == C1)
    fig.suptitle(f"One NCRL fine-tuning update on an RTX 4070 = {total / 1000:.2f} s", x=0.01, ha="left",
                 fontsize=11, color=INK)
    ax.set_title(f"world-model update {wm_sum / total:.0%}  |  imagination + actor-critic {1 - wm_sum / total:.0%}",
                 loc="left", fontsize=9.5, color=INK2)
    ax.set_xlim(0, max(v for _, v, _ in rows) * 1.18)
    style(ax)
    h = [plt.Rectangle((0, 0), 1, 1, color=C1), plt.Rectangle((0, 0), 1, 1, color=C2)]
    ax.legend(h, ["world-model update", "imagination + actor-critic"], loc="lower right")
    fig.text(0.01, 0.01, "* includes recomputing the checkpointed imagination (needed to fit 12 GB; results identical). "
             "Paper size, batch 16x64, fp32.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, "update_time.png"), dpi=160)
    plt.close(fig)


def wm_error():
    r = json.load(open(os.path.join(ROOT, "results", "diag_wm_error.json")))["error"]
    models = [("pre", "pretrained"), ("ft20", "fine-tuned 20k frames"), ("ft40", "fine-tuned 40k frames")]
    sets = [("on_rl", "online episodes already trained on", C1), ("on_new", "next online episodes (not yet trained on)", C2),
            ("prior", "retrieved offline data (rehearsed)", C3)]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    w = 0.22
    for j, (k, lab, c) in enumerate(sets):
        xs = [i + (j - 1) * (w + 0.03) for i in range(len(models))]
        vals = [r[m][k]["openloop"]["mean"] for m, _ in models]
        ses = [r[m][k]["openloop"]["se"] for m, _ in models]
        ax.bar(xs, vals, width=w, color=c, label=lab, yerr=ses, ecolor=INK2, error_kw=dict(lw=0.8, capsize=2))
        if k in ("on_rl", "on_new"):
            ax.text(xs[2], vals[2] + ses[2] + 2.5, f"{vals[2]:.1f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(models)), [n for _, n in models], color=INK)
    ax.set_ylabel("open-loop image error (32 imagined steps)")
    ax.set_title("Fine-tuning mostly fits data already in the replay buffer (Quadruped Walk, seed 1, mean ± SE, 96 chunks)",
                 loc="left", fontsize=10)
    style(ax, "y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wm_error.png"), dpi=160)
    plt.close(fig)


def pilot():
    import summarize_pilot as S
    arms = {}
    if os.path.exists(os.path.join(S.REF, "branch_log.jsonl")):
        arms["R1 (full)"] = S.read_arm(S.REF)
    for name in sorted(os.listdir(S.RUNS)):
        p = os.path.join(S.RUNS, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "branch_log.jsonl")) and name != "smoke":
            a = S.read_arm(p)
            if a["done"]:
                arms[name] = a
    points = sorted({a["start_frame"] for a in arms.values()})
    fig, axes = plt.subplots(1, len(points), figsize=(4.0 * len(points) + 0.6, 3.8), sharey=True, squeeze=False)
    for ax, bp in zip(axes[0], points):
        group = {k: v for k, v in arms.items() if v["start_frame"] == bp and v["evals"].get(bp + 20000) is not None}
        fulls = [v["evals"][bp + 20000] for k, v in group.items() if k in S.FULL_ARMS or k == "R1 (full)"]
        if len(fulls) >= 2:
            ax.axhspan(min(fulls), max(fulls), color=C1, alpha=0.10, lw=0)
        for k, v in group.items():
            full = k in S.FULL_ARMS or k == "R1 (full)"
            ax.scatter(v["update_s"], v["evals"][bp + 20000], s=60, color=C1 if full else C2, edgecolor=SURF,
                       linewidth=2, zorder=3)
            lab = k + (" (old off: WD 0.25%)" if v["args"].get("wm_every") == 0 and not v["args"].get("strict_off") else "")
            ax.annotate(lab, (v["update_s"], v["evals"][bp + 20000]), xytext=(6, -3), textcoords="offset points",
                        fontsize=8.5, color=INK)
        ax.set_title(f"branch at {bp // 1000}k -> eval at {(bp + 20000) // 1000}k frames", loc="left", fontsize=10)
        ax.set_xlabel("update seconds per epoch (125 updates)")
        style(ax, "both")
        ax.set_xlim(220, 400)
    allv = [v["evals"][v["start_frame"] + 20000] for v in arms.values() if v["evals"].get(v["start_frame"] + 20000)]
    axes[0][0].set_ylim(min(allv) - 60, max(allv) + 40)
    axes[0][0].set_ylabel("eval return (10 episodes)")
    h = [plt.Line2D([], [], marker="o", ls="", color=C1, ms=7), plt.Line2D([], [], marker="o", ls="", color=C2, ms=7),
         plt.Rectangle((0, 0), 1, 1, color=C1, alpha=0.10)]
    fig.legend(h, ["original update (full WM)", "fewer / no WM updates", "spread of full-update replicates"],
               loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Branch pilot (Quadruped Walk, 1 branch seed): compute vs return — in progress", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(OUT, "pilot_tradeoff.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    update_time(); wm_error(); pilot()
    print("figures ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
