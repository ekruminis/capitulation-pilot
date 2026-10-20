"""Draw the pilot's two figures from the CSVs that `analyse.py all` writes.

  results/fig_turns.png    share of runs not yet flipped at any turn, after each turn
  results/fig_effects.png  the effects with their 95% intervals

Run:  uv run python plots.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import prompts

# Colours and markers follow the condition, never its rank, so a condition
# looks the same in every panel and figure.
STYLE = {
    "reask": ("#2a78d6", "o", "Re-ask"),
    "one_author": ("#eb6834", "s", "One author"),
    "three_authors": ("#1baf7a", "^", "Three authors"),
    "anonymous": ("#c98500", "D", "Anonymous note"),
}
ARM_TITLE = {"A": "Arm A: push to approve a backdoor", "B": "Arm B: push to reject clean code"}
ARM_SHORT = {"A": "Arm A: push to approve", "B": "Arm B: push to reject"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    }
)


def results_dir() -> Path:
    return Path(os.environ.get("PILOT_RESULTS_DIR", "results"))


def plot_turns(turns: pd.DataFrame, path: Path) -> None:
    arms = [a for a in ("A", "B") if a in set(turns.arm)]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.8), sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms, strict=True):
        sub = turns[turns.arm == arm]
        ends = []
        for cond in prompts.CONDITIONS:
            line = sub[sub.condition == cond].sort_values("turn")
            if line.empty:
                continue
            colour, marker, label = STYLE[cond]
            ax.plot(line.turn, 100 * line.unflipped, color=colour, lw=2, marker=marker, ms=6, label=label)
            ends.append((100 * line.unflipped.iloc[-1], label))
        # Direct labels at the right end, nudged apart so they don't collide.
        ends.sort()
        last_y = -100.0
        for y, label in ends:
            y = max(y, last_y + 5)
            ax.annotate(label, (3, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
            last_y = y
        ax.set_title(ARM_TITLE[arm], loc="left", fontsize=10.5, color=INK)
        ax.set_xticks([0, 1, 2, 3], ["Own verdict", "Turn 1", "Turn 2", "Turn 3"])
        ax.set_xlim(-0.2, 3.9)
        ax.set_ylim(0, 102)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Runs not yet flipped (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


EFFECT_LABEL = {
    "capitulation": "Capitulation\n(one author - re-ask)",
    "crowd": "Crowd\n(three authors - one)",
    "stake": "Stake\n(one author - anonymous)",
}


def plot_effects(effects: pd.DataFrame, path: Path) -> None:
    names = [n for n in EFFECT_LABEL if n in set(effects.effect)]
    arm_style = {"A": ("#2a78d6", "o", -0.12), "B": ("#eb6834", "s", 0.12)}
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.9 * len(names)))
    for arm, (colour, marker, offset) in arm_style.items():
        sub = effects[effects.arm == arm].set_index("effect")
        for i, name in enumerate(names):
            if name not in sub.index:
                continue
            r = sub.loc[name]
            y = i + offset
            ax.plot([100 * r.lo, 100 * r.hi], [y, y], color=colour, lw=2, solid_capstyle="round")
            ax.plot(100 * r.estimate, y, marker=marker, ms=7, color=colour, mec="#fcfcfb", mew=1.5,
                    label=ARM_SHORT[arm] if i == 0 else None)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(names)), [EFFECT_LABEL[n] for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Difference in flip rate (percentage points), 95% interval")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    out = results_dir()
    turns = pd.read_csv(out / "turns.csv")
    effects = pd.read_csv(out / "effects.csv")
    plot_turns(turns, out / "fig_turns.png")
    plot_effects(effects, out / "fig_effects.png")
    print(f"Saved {out / 'fig_turns.png'} and {out / 'fig_effects.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
