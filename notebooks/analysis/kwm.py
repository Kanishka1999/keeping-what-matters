"""Shared loading, fitting and chart style for the analysis notebooks.
Colours are the validated categorical slots 1-4 (CVD-safe; two sit below 3:1 contrast, so every chart
also has direct labels, distinct line styles/markers and a table of the exact numbers)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2"
SKILLS = ("instruction_prompt", "maths", "tools_correct", "charts")
SE_KEY = {"instruction_prompt": "instruction_se", "maths": "maths_se", "tools_correct": "tools_se", "charts": "charts_se"}

def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)

def exists(rel):
    return os.path.exists(os.path.join(RES, rel))

def lost(r, key):
    """Percent of the starting refusal margin lost (0 = untouched)."""
    return 100 * (1 - r[key])

def decay_fit(tokens, y):
    """Least-squares fit of y = a*log2(tokens) + b; returns (a, b, R^2)."""
    x, y = np.log2(np.asarray(tokens, float)), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a * x + b)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return a, b, r2

def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6, "figure.dpi": 110})

def tok_axis(ax, ticks):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks)
    ax.set_xticklabels([f"{round(t / 1e6, 1):g}M" if t >= 1e6 else f"{t / 1e3:.0f}k" for t in ticks]); ax.minorticks_off()

def end_label(ax, x, y, text):
    """Direct label at a series' last point, in ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)

def end_labels(ax, x, items, gap):
    """Direct labels for several series ending at the same x, nudged apart vertically by at least `gap` (data units)."""
    items = sorted(items); ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = max(0.0, (ys[-1] - items[-1][0]) / 2) if len(ys) > 1 else 0.0
    for (_, text), y in zip(items, ys):
        end_label(ax, x, y - shift, text)
