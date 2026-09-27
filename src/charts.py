"""Static chart images for the video (dark theme, 1920-wide)."""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]
BG = "#111110"; INK = "#ffffff"; INK2 = "#c3c2b7"; MUTED = "#8a8980"; GRID = "#2c2c2a"
C = {"real": "#3987e5", "shuf0": "#d95926", "noconn": "#199e70", "real_rewired_after_training": "#6f6e69", "random": "#4a4a47"}
for f in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Arial", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": BG, "axes.facecolor": BG})
rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def a1_metrics():
    SC = [("defend_the_center", "defend the center", "kills per game", "kills", 1),
          ("defend_the_line", "defend the line", "kills per game", "kills", 1),
          ("health_gathering", "health gathering", "seconds survived", "ret", 1 / 35),
          ("basic", "basic", "score", "ret", 1),
          ("deadly_corridor", "deadly corridor", "score", "ret", 1)]
    arms = ["real", "shuf0", "noconn", "real_rewired_after_training", "random"]
    d = collections.defaultdict(list)
    for arm in arms:
        p = ROOT / f"results/a1/{arm}.jsonl"
        for l in open(p):
            r = json.loads(l); d[(arm, r["scenario"])].append(r)
    res = {}
    for key, title, unit, field, scale in SC:
        for arm in arms:
            x = np.array([r[field] for r in d[(arm, key)]], float) * scale
            if len(x): res[(key, arm)] = (x.mean(), *boot(x), len(x))
    return SC, arms, res

LABEL = {"real": "real wiring", "shuf0": "wiring shuffled", "noconn": "no connectome",
         "real_rewired_after_training": "real, rewired after training", "random": "random buttons"}

def chart_a1(path, show_refs=True):
    SC, arms, res = a1_metrics()
    arms_shown = arms if show_refs else arms[:3]
    fig, axes = plt.subplots(1, 5, figsize=(19.2, 6.0), dpi=100)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.14, wspace=0.28)
    for ax, (key, title, unit, field, scale) in zip(axes, SC):
        ys = np.arange(len(arms_shown))[::-1]
        for y, arm in zip(ys, arms_shown):
            if (key, arm) not in res: continue
            m, lo, hi, n = res[(key, arm)]
            ax.barh(y, m, height=0.62, color=C[arm], zorder=2)
            ax.plot([lo, hi], [y, y], color=INK, lw=2, zorder=3, solid_capstyle="butt")
            ax.text(max(m, 0) + 0.02 * ax.get_xlim()[1], y, "", va="center")
        top = max(res[(key, a)][2] for a in arms_shown if (key, a) in res)
        lo_ = min(0, min(res[(key, a)][1] for a in arms_shown if (key, a) in res))
        ax.set_xlim(lo_ * 1.35, top * 1.18)
        for y, arm in zip(ys, arms_shown):
            if (key, arm) in res:
                m = res[(key, arm)][0]
                xt = max(m, res[(key, arm)][2]) + 0.03 * (top * 1.18 - lo_) if m >= 0 else res[(key, arm)][1] - 0.03 * (top * 1.18 - lo_)
                ax.text(xt, y, f"{m:.1f}" if abs(m) < 100 else f"{m:.0f}",
                        va="center", ha="left" if m >= 0 else "right", fontsize=15, color=INK)
        ax.set_title(title, fontsize=20, color=INK, pad=10, loc="left")
        ax.set_xlabel(unit, fontsize=14)
        ax.set_yticks(ys); ax.set_yticklabels([LABEL[a] for a in arms_shown] if ax is axes[0] else [], fontsize=16)
        ax.grid(axis="x", color=GRID, lw=1, zorder=0); ax.tick_params(axis="x", labelsize=12)
        for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
        ax.axvline(0, color=MUTED, lw=1)
    fig.savefig(path, facecolor=BG); plt.close(fig)
    return res

def chart_b(path):
    import pandas as pd
    ids = json.loads((ROOT / "src/partB_ids.json").read_text())
    conds = [("real", "real wiring", "#3987e5")] + [(f"shuf{i}", f"shuffle seed {i}", "#d95926") for i in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 6.0), dpi=100, gridspec_kw={"width_ratios": [2.3, 1]})
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.14, wspace=0.30)
    ax = axes[0]; y0 = 0; ticks = []; rates = []
    for c, lab, col in conds:
        p = ROOT / f"partB/spikes/sugar_100Hz_{c}.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p); m = df.loc[df.flywire_id == ids["mn9"][0]]
        for t in range(30):
            ts = m.loc[m.trial == t, "t"].to_numpy() * 1000
            ax.vlines(ts, y0 + t, y0 + t + 0.8, color=col, lw=1.2)
        per = np.bincount(m.trial.astype(int), minlength=30)
        if per.sum() == 0: ax.text(500, y0 + 15, "no spikes in 30 trials", ha="center", va="center", fontsize=18, color=MUTED)
        rates.append((lab, per.mean(), *boot(per), col))
        ticks.append((y0 + 15, lab)); y0 += 34
    ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=16)
    ax.set_ylim(y0 - 2, -3); ax.set_xlim(0, 1000); ax.set_xlabel("time (ms), sugar neurons stimulated the whole trial", fontsize=14)
    ax.set_title("MN9 (proboscis motor neuron) spikes, 30 trials each", fontsize=20, loc="left", pad=10)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax2 = axes[1]
    for i, (lab, m, lo, hi, col) in enumerate(rates):
        ax2.barh(len(rates) - 1 - i, m, color=col, height=0.62, zorder=2)
        ax2.plot([lo, hi], [len(rates) - 1 - i] * 2, color=INK, lw=2, zorder=3)
        ax2.text(hi + 2, len(rates) - 1 - i, f"{m:.1f} Hz", va="center", fontsize=16)
    ax2.set_yticks(range(len(rates))); ax2.set_yticklabels([r[0] for r in rates][::-1], fontsize=14)
    ax2.set_xlim(0, max(r[3] for r in rates) * 1.35 + 1); ax2.set_xlabel("MN9 firing rate (Hz)", fontsize=14)
    ax2.set_title("mean, 95% CI", fontsize=20, loc="left", pad=10); ax2.grid(axis="x", color=GRID, zorder=0)
    for s in ["top", "right", "left"]: ax2.spines[s].set_visible(False)
    fig.savefig(path, facecolor=BG); plt.close(fig)
    return rates

if __name__ == "__main__":
    import sys
    if "a1" in sys.argv: print(chart_a1(ROOT / "video/chart_a1.png"))
    if "b" in sys.argv: print(chart_b(ROOT / "video/chart_b.png"))

def chart_a2(path):
    s = json.loads((ROOT / "results/a2/summary.json").read_text())
    rows = [("real", "real wiring", "#3987e5"), ("shuffled (3 seeds pooled)", "wiring shuffled (3 seeds)", "#d95926"),
            ("blind", "real wiring, eyes off", "#6f6e69"), ("noconn", "no connections", "#4a4a47"),
            ("spray_matched", "no brain: fixed autopilot", "#199e70")]
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 5.6), dpi=100)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.86, bottom=0.15, wspace=0.12)
    for ax, (key, ci, title, unit) in zip(axes, [("survival_mean", "survival_ci", "survival per round", "seconds (capped at 60)"),
                                                ("kills_mean", "kills_ci", "kills per round", "kills")]):
        ys = np.arange(len(rows))[::-1]
        for y, (arm, lab, col) in zip(ys, rows):
            m = s[arm][key]; lo, hi = s[arm][ci]
            ax.barh(y, m, color=col, height=0.62, zorder=2)
            ax.plot([lo, hi], [y, y], color=INK, lw=2, zorder=3)
            ax.text(hi + 0.02 * ax.get_xlim()[1] + (1.2 if key == "survival_mean" else 0.3), y, f"{m:.1f}", va="center", fontsize=16)
        top = max(s[a][ci][1] for a, _, _ in rows); ax.set_xlim(0, top * 1.2)
        ax.set_yticks(ys); ax.set_yticklabels([f"{r[1]}  (n={s[r[0]]['n']})" for r in rows] if ax is axes[0] else [], fontsize=16)
        ax.set_title(title, fontsize=20, loc="left", pad=10); ax.set_xlabel(unit, fontsize=14)
        ax.grid(axis="x", color=GRID, zorder=0)
        for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    fig.savefig(path, facecolor=BG); plt.close(fig)

if __name__ == "__main__":
    import sys
    if "a2" in sys.argv: chart_a2(ROOT / "video/chart_a2.png")
