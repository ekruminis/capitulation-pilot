"""Draw the pilot's figures from the CSVs that `analyse.py stability` and `all` write.

  results/fig_stability.png  Stage A: items held stably right, not stable, or stably wrong
  results/fig_verdicts.png   arm B: share of runs saying REJECT after each message, by condition
  results/fig_reasons.png    arm B: flipped runs per condition, by the reason the monitor gave

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


def plot_verdicts_by_turn(runs: pd.DataFrame, path: Path) -> None:
    """Arm B: share of runs saying REJECT after each message, by condition."""
    sub = runs[runs.arm == "B"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for cond in prompts.CONDITIONS:
        rows = sub[sub.condition == cond]
        if rows.empty:
            continue
        share = [100 * rows[f"v{t}"].eq("REJECT").mean() for t in range(4)]
        colour, marker, label = STYLE[cond]
        ax.plot(range(4), share, color=colour, lw=2, marker=marker, ms=7, mec="#fcfcfb", mew=1.2, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5, handlelength=2.2)
    ax.set_xticks(range(4), ["Own verdict", "After message 1", "After message 2", "After message 3"])
    ax.set_xlim(-0.2, 3.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Runs saying REJECT")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    n = len(sub) // max(sub.condition.nunique(), 1)
    ax.set_title("Clean code the monitor had approved: how often it said REJECT after each message",
                 loc="left", fontsize=11, color=INK, pad=30)
    ax.text(0, -0.16, f"{n} runs per condition (36 clean items x 3). Pushed the other way, to approve 60 backdoors "
            "it had rejected,\nthe monitor never said APPROVE, in any condition or turn.",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# What the monitor said when it flipped, from the notes column of the coded sheet.
REASONS = [
    ("format", "#7b5ea7", "Invented a bug from the \"legacy format\" claim"),
    ("logic", "#3d5a80", "Named an unrelated logic bug"),
    ("other", "#d9d8d1", "Other (accepted the claim as stated, or rejected to be safe)"),
]


def flip_reason(notes: str) -> str:
    notes = str(notes)
    if "no code" in notes:
        return "other"
    if "logic bug" in notes:
        return "logic"
    if "input-format" in notes:
        return "format"
    return "other"


def plot_flip_reasons(coded: pd.DataFrame, key: pd.DataFrame, path: Path) -> None:
    """Arm B: number of flipped runs per condition, split by the reason the monitor gave."""
    d = coded.merge(key[["code_id", "condition"]], on="code_id")
    d["reason"] = d.notes.fillna("").map(flip_reason)
    counts = pd.crosstab(d.condition, d.reason).reindex(index=prompts.CONDITIONS, columns=[r for r, *_ in REASONS],
                                                        fill_value=0)
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for y, cond in enumerate(prompts.CONDITIONS):
        left = 0
        for reason, colour, _ in REASONS:
            n = int(counts.loc[cond, reason])
            if n == 0:
                continue
            ax.barh(y, n, left=left, height=0.62, color=colour, edgecolor="#fcfcfb", linewidth=2)
            if n >= 4:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9,
                        color="#ffffff" if reason != "other" else INK)
            left += n
        ax.text(left + 1.2, y, f"{left} of 108", va="center", ha="left", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(prompts.CONDITIONS)), [STYLE[c][2] for c in prompts.CONDITIONS], color=INK)
    ax.set_ylim(len(prompts.CONDITIONS) - 0.5, -0.5)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Runs that flipped to REJECT at some turn")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in REASONS]
    ax.legend(handles, [name for *_, name in REASONS], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1,
              frameon=False, fontsize=9, handlelength=1.2)
    ax.set_title("What the monitor said when it flipped on clean code", loc="left", fontsize=11, color=INK, pad=62)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Stage A outcome per item: held the right verdict, wavered, or held the wrong one.
OUTCOME_STYLE = [
    ("right", "#2a78d6", "#ffffff", "Stable, right verdict"),
    ("mixed", "#d9d8d1", INK, "Not stable"),
    ("wrong", "#eb6834", INK, "Stable, wrong verdict"),
]


def plot_stability(items: pd.DataFrame, path: Path) -> None:
    """Stage A: how many items the monitor judged the same way at least 8 times in 9."""
    rows = [
        ("backdoor", "Backdoored code", "REJECT", "rejected"),
        ("clean", "Clean code", "APPROVE", "approved"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    for y, (label, title, right_verdict, right_word) in enumerate(rows):
        sub = items[items.label == label]
        counts = {
            "right": int((sub.stable & sub.majority.eq(right_verdict)).sum()),
            "mixed": int((~sub.stable).sum()),
            "wrong": int((sub.stable & ~sub.majority.eq(right_verdict)).sum()),
        }
        left = 0
        for key, colour, text_colour, _ in OUTCOME_STYLE:
            n = counts[key]
            share = 100 * n / len(sub)
            ax.barh(y, share, left=left, height=0.6, color=colour, edgecolor="#fcfcfb", linewidth=2)
            if key == "right":
                ax.text(left + 1.5, y, f"{n} stably {right_word}", va="center", ha="left", color=text_colour, fontsize=10)
            left += share
        # The small segments are labelled once, to the right of the bar.
        ax.text(101.5, y, f"{counts['mixed']} not stable\n{counts['wrong']} stably wrong", va="center", ha="left",
                color=MUTED, fontsize=9)
        ax.text(-1.5, y, f"{title}\n({len(sub)} items)", va="center", ha="right", color=INK, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(1.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in OUTCOME_STYLE]
    ax.legend(handles, [name for *_, name in OUTCOME_STYLE], loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
              frameon=False, fontsize=9, handlelength=1.2)
    ax.set_title("Does the monitor give the same verdict each time it reviews an item?", loc="left", pad=34,
                 fontsize=11, color=INK)
    ax.text(0, -0.2, "Each item was reviewed 9 times. Stable: the same verdict at least 8 times. "
            "Stage B uses the blue items (arm A: 60 of the 91; arm B: all 36).",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    out = results_dir()
    saved = []
    if (out / "stability_items.csv").exists():
        plot_stability(pd.read_csv(out / "stability_items.csv"), out / "fig_stability.png")
        saved.append(out / "fig_stability.png")
    if (out / "pressure_runs.csv").exists():
        runs = pd.read_csv(out / "pressure_runs.csv")
        if "B" in set(runs.arm):
            plot_verdicts_by_turn(runs, out / "fig_verdicts.png")
            saved.append(out / "fig_verdicts.png")
    if (out / "flips_to_code_B.csv").exists() and (out / "flips_key_B.csv").exists():
        plot_flip_reasons(pd.read_csv(out / "flips_to_code_B.csv"), pd.read_csv(out / "flips_key_B.csv"),
                          out / "fig_reasons.png")
        saved.append(out / "fig_reasons.png")
    print("Saved " + ", ".join(str(p) for p in saved) if saved else "Nothing to plot yet.")


if __name__ == "__main__":
    main(sys.argv[1:])
