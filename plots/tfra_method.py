"""Monochrome vector diagram of TFRA operations and readout schedules."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


OUT = Path(__file__).resolve().parent / "outputs"


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "text.color": "black",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    # Keep the label sizes and full manuscript width, but compress vertical
    # spacing so the diagram and its caption can share a page with body text.
    fig, ax = plt.subplots(figsize=(13.6, 7.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 13.6), ylim=(0, 9.6))
    ax.axis("off")
    labels = []
    offset = 3.4

    def text(x, y, s, size=12, ha="center", weight="normal"):
        t = ax.text(x, y + offset, s, fontsize=size, ha=ha, va="center",
                    weight=weight, linespacing=1.35)
        labels.append(t)
        return t

    def arrow(start, end, dashed=False):
        start = (start[0], start[1] + offset)
        end = (end[0], end[1] + offset)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=12, linewidth=0.9,
                                    linestyle=(0, (3, 3)) if dashed else "-",
                                    color="black", shrinkA=0, shrinkB=0))

    text(0.35, 5.94, "(a) Shared adaptation procedure", size=14,
         ha="left", weight="bold")
    for x, s in [(1.20, "Source checkpoint"), (4.17, "Initial fit"),
                 (7.70, "Temporal adaptation"), (11.42, "Post-update refit")]:
        text(x, 5.45, s, size=13, weight="bold")

    text(1.20, 4.83, r"$\theta_S$", size=23)
    text(1.20, 4.22, "Pretrained encoder", size=12)

    text(4.17, 4.99, r"$(w_0,b_0)=\arg\min_{w,b}$", size=16)
    text(4.17, 4.45, r"$\mathcal{J}_{D_{\mathrm{init}}}(\theta_S,w,b)$", size=16)
    text(4.17, 4.02, "Encoder fixed; before first update", size=12)
    text(4.17, 3.72, r"$D_{\mathrm{init}}=B_1$ for B; $D_1$ otherwise", size=11)

    text(7.70, 4.86, r"$\theta_t\ \longrightarrow\ \theta_{t+1}$", size=18)
    # A time axis indicates the batch order, without inventing feature trajectories.
    arrow((6.58, 4.36), (8.85, 4.36))
    for x, s in [(6.70, r"$B_1$"), (7.23, r"$B_2$"),
                 (7.76, r"$B_3$"), (8.35, r"$\cdots$")]:
        ax.plot([x, x], [4.30 + offset, 4.42 + offset], color="black", lw=0.7)
        text(x, 4.08, s, size=12)
    text(7.70, 3.72, "Chronological batches; readout rule in (b)", size=11)

    text(11.42, 4.99, r"$(w_e^{\mathrm{eval}},b_e^{\mathrm{eval}})$", size=16)
    text(11.42, 4.45, r"$=\arg\min_{w,b}\mathcal{J}_{D_e}(\theta',w,b)$", size=15)
    text(11.42, 4.02, r"Updated encoder $\theta'$ fixed; full Train", size=12)
    text(11.42, 3.72, "At each epoch end, for evaluation", size=11)

    for start, end in [(1.95, 2.77), (5.54, 6.33), (9.03, 9.63)]:
        arrow((start, 4.83), (end, 4.83))

    text(6.80, 3.24,
         r"Shared: fixed hidden map $A_0,c_0$; ridge fitting on current features; MSE encoder updates.",
         size=12)

    offset = 0.0
    text(0.35, 6.18, "(b) Readout states across epochs", size=14,
         ha="left", weight="bold")
    text(13.25, 6.18, r"$r=(w,b)$: final linear readout", size=12, ha="right")
    for left, right, title in [(2.80, 7.50, "Epoch 1"), (10.38, 13.03, "Epoch 2")]:
        ax.plot([left, left, right, right], [5.63, 5.70, 5.70, 5.63],
                color="black", lw=0.7)
        text((left + right) / 2, 5.88, title, size=12)

    def state(x, y, value, marker="o"):
        ax.scatter([x], [y], s=850 if marker == "D" else 650,
                   marker=marker, facecolors="white",
                   edgecolors="black", linewidths=0.9, zorder=4)
        text(x, y, value, size=14)
        labels[-1].set_zorder(5)

    # These are discrete parameter states, not loss or feature-space trajectories.
    # Each epoch-end fit is a separate evaluation branch. The training path
    # preserves/restores the prior head for Fixed and Joint.
    lanes = [
        (5.12, "TFRA-E", [r"$r_0$", r"$r_0$", r"$r_0$", r"$r_2^\star$", r"$r_2^\star$"],
         r"$\theta$", "fit on new candidates", ["D", "o", "o", "D", "o"]),
        (3.80, "TFRA-B", [r"$r_1^\star$", r"$r_2^\star$", r"$r_2^\star$", r"$r_3^\star$", r"$r_3^\star$"],
         r"$\theta$", "fit next batch", ["s", "s", "o", "s", "o"]),
        (2.40, "TFRA-Fixed", [r"$r_0$"] * 5,
         r"$\theta$", r"restore $r_0$", ["D", "o", "o", "o", "o"]),
        (1.08, "TFRA-Joint", [r"$r_0$", r"$r_a$", r"$r_b$", r"$r_b$", r"$r_c$"],
         r"$\theta,r$", r"restore $r_b$", ["D", "o", "o", "o", "o"]),
    ]
    xs = [3.0, 5.2, 7.3, 10.6, 12.8]
    for y, method, values, update, continuation, markers in lanes:
        text(0.45, y, method, size=13, ha="left")
        for i, (x, value) in enumerate(zip(xs, values)):
            state(x, y, value, marker=markers[i])
        for i in (0, 1, 3):
            arrow((xs[i] + 0.25, y), (xs[i + 1] - 0.25, y))
            label = "update " + update
            if method == "TFRA-B" and i == 0:
                label += r"; fit $B_2$"
            text((xs[i] + xs[i + 1]) / 2, y + 0.20, label, size=11)
        arrow((7.55, y), (10.35, y))
        text(8.95, y + 0.20, continuation, size=11)

        # Refit current features for evaluation; show this off the training path.
        ax.plot([7.3, 7.3, 8.15], [y - 0.20, y - 0.59, y - 0.59],
                color="black", lw=0.85, ls=(0, (3, 3)))
        arrow((8.15, y - 0.59), (8.39, y - 0.59), dashed=True)
        ax.scatter([8.52], [y - 0.59], s=110, marker="D", facecolors="white",
                   edgecolors="black", linewidths=0.9, zorder=4)
        text(8.52, y - 0.36, "refit", size=10.5)
        arrow((8.65, y - 0.59), (9.20, y - 0.59), dashed=True)
        text(9.32, y - 0.59, "evaluate", size=11, ha="left")
        arrow((13.06, y), (13.32, y))

    ax.scatter([0.57], [0.32], s=90, marker="D", facecolors="white", edgecolors="black", linewidths=0.9)
    text(0.78, 0.32, "full-Train ridge fit", size=10.5, ha="left")
    ax.scatter([3.05], [0.32], s=80, marker="s", facecolors="white", edgecolors="black", linewidths=0.9)
    text(3.26, 0.32, "batch ridge fit", size=10.5, ha="left")
    ax.scatter([5.10], [0.32], s=80, marker="o", facecolors="white", edgecolors="black", linewidths=0.9)
    text(5.31, 0.32, "training readout state", size=10.5, ha="left")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [t.get_window_extent(renderer) for t in labels]
    for t, b in zip(labels, bounds):
        assert fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1), t.get_text()
    for i, a in enumerate(bounds):
        for j in range(i):
            assert not a.overlaps(bounds[j]), (labels[i].get_text(), labels[j].get_text())

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"tfra_method.{ext}", dpi=240,
                    bbox_inches="tight", pad_inches=0.08, facecolor="white")
    svg = OUT / "tfra_method.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
