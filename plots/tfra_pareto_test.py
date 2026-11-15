"""Plot observed Test means; uncertainty is reported in the readout ablation table."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "finetune": ("FT", "Fine-tuning", "#555555", "o"),
    "frozen": ("Frozen", "Frozen Encoder", "#8A7B9E", "^"),
    "warmstart": ("Warm-start", "Ridge Warm-start", "#B16B36", "s"),
    "joint": ("Joint MSE", "Joint MSE", "#8C8650", "v"),
    "tfra": ("TFRA-E", "TFRA-E", "#39658C", "D"),
    "batch": ("TFRA-B", "TFRA-B", "#B44957", "P"),
    "initial_fixed": ("TFRA-Fixed", "TFRA-Fixed", "#308773", "<"),
    "joint_init": ("TFRA-Joint", "TFRA-Joint", "#42643C", ">"),
}


def render(panel, *, output_dir=None):
    assert set(panel) <= set(STYLES)
    styles = {method: style for method, style in STYLES.items() if method in panel}
    # Match the typography, method colors, and markers of the budget figure.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.7))
    fig.subplots_adjust(left=.065, right=.985, bottom=.30, top=.87, wspace=.38)
    fields = [("D_G", "ap", False, True, "(a) Test prediction and damage"),
              ("B_G", "ap", True, True, "(b) Test prediction and benefit"),
              ("D_G", "B_G", False, True, "(c) Test damage and benefit")]
    labels = {"D_G": r"$D_G$ (AP percentage points) ↓",
              "B_G": r"$B_G$ (AP percentage points) ↑", "ap": "Test AP (%) ↑"}
    offsets = [
        {"initial_fixed": (-2, -24), "joint_init": (-13, 19), "tfra": (6, 8),
         "joint": (6, -12), "finetune": (-12, -14), "warmstart": (6, -7)},
        {"initial_fixed": (-52, -22), "joint_init": (-42, 18), "tfra": (7, -4),
         "joint": (-34, -9), "finetune": (6, -13), "warmstart": (6, -7)},
        {"initial_fixed": (-5, -23), "joint_init": (-8, 19), "tfra": (6, 7),
         "joint": (-24, 8), "finetune": (-16, -15)},
    ]
    for index, (ax, (x, y, xmax, ymax, title)) in enumerate(zip(axes, fields)):
        def dominates(a, b):
            av = (a[x] if xmax else -a[x], a[y] if ymax else -a[y])
            bv = (b[x] if xmax else -b[x], b[y] if ymax else -b[y])
            return all(v >= w for v, w in zip(av, bv)) and any(v > w for v, w in zip(av, bv))

        rows = list(panel.values())
        front = sorted([r for r in rows if not any(dominates(a, r) for a in rows)], key=lambda r: r[x])
        if len(front) > 1:
            ax.plot([100 * r[x] for r in front], [100 * r[y] for r in front], ":",
                    color=".65", lw=.9)
        for method, (short, _, color, marker) in styles.items():
            row = panel[method]
            point = 100 * row[x], 100 * row[y]
            ax.scatter(*point, s=32, c=color, marker=marker,
                       edgecolors="white", linewidths=.4, zorder=4)
            offset = offsets[index].get(method, (6, 7 if method == "batch" else -10))
            ax.annotate(short, point, xytext=offset, textcoords="offset points", fontsize=7.5,
                        color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=.55, shrinkA=2, shrinkB=4)
                        if method in {"initial_fixed", "joint_init", "tfra"} else None)
        if x == "D_G" and all(r[x] > 0 for r in rows):
            ax.set_xscale("log")
        ax.margins(x=.38, y=.30)
        ax.set_title(title, loc="left", pad=11)
        ax.set_xlabel(labels[x])
        ax.set_ylabel(labels[y])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".9", lw=.6)
        ax.tick_params(width=.65, length=3)
        ax.set_axisbelow(True)

    handles = [Line2D([0], [0], color=color, marker=marker, linestyle="none", markersize=5,
                      label=name) for _, name, color, marker in styles.values()]
    # Two semantic rows, with the same TFRA order as the budget figure.
    order = [i for pair in zip(range(4), range(4, 8)) for i in pair] if len(handles) == 8 else range(len(handles))
    fig.legend(handles=[handles[i] for i in order], loc="lower center", bbox_to_anchor=(.5, .065),
               ncol=4, frameon=False, columnspacing=2, handletextpad=.5)
    fig.legend(handles=[Line2D([0], [0], color=".65", ls=":", lw=.9,
                              label="Pairwise nondominated means (visual guide)")],
               loc="lower center", bbox_to_anchor=(.5, .005), frameon=False)
    dest = Path(output_dir) if output_dir is not None else ROOT / "plots/outputs"
    dest.mkdir(exist_ok=True, parents=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(dest / f"tfra_pareto_test.{ext}", dpi=220)
    svg = dest / "tfra_pareto_test.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
