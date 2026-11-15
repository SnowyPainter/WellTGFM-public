"""Seed interaction diagnostics, replacing the two tables in claim.md §5.2.3.

Values transcribed from the original manuscript tables: target means of eight
3x3 RMS decompositions (rounded to 0.001 AP pp), and the MOOC->LastFM
Validation AP matrix (six decimals). No synthetic observations or error bars.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['Wikipedia', 'Reddit', 'MOOC', 'LastFM']
RMS = np.array([[1.113, .655, 1.309], [.458, .108, .232],
                [.234, .208, .304], [.574, .528, .928]])
AP = np.array([[.598257, .711330, .722232],
               [.720924, .718510, .719007],
               [.697714, .706228, .635923]])


def main():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .65})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.76, wspace=.32)
    colors = ['#39658C', '#B16B36', '#567B58']
    ax = axes[0]
    x = np.arange(4)
    for i, (name, hatch) in enumerate(zip(['Pretraining', 'Adaptation', 'Interaction'], ['', '//', '..'])):
        ax.bar(x+(i-1)*.24, RMS[:, i], width=.22, label=name, color=colors[i],
               edgecolor='white', linewidth=.5, hatch=hatch, zorder=3)
    ax.set(xticks=x, xticklabels=TARGETS, ylabel='Effect RMS (AP percentage points)',
           ylim=(0, 1.45), yticks=np.arange(0, 1.41, .35))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title('(a) Effects across targets', loc='left', pad=40)
    ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.01),
              columnspacing=.8, handlelength=1.1, handletextpad=.4)
    ax = axes[1]
    # Seeds are categories; connecting lines visualize interactions, not interpolation.
    for i, (seed, marker, ls) in enumerate(zip([7,17,37], ['o','s','^'], ['-','--',':'])):
        ax.plot(np.arange(3), AP[i], label=str(seed), color=colors[i], marker=marker,
                linestyle=ls, lw=1.4, ms=5, zorder=3, markeredgecolor='white', markeredgewidth=.5)
    ax.set(xticks=[0,1,2], xticklabels=['7','17','37'], xlabel='Adaptation seed',
           ylabel='Validation AP', xlim=(-.13, 2.13), ylim=(.58,.745),
           yticks=[.60,.64,.68,.72])
    ax.set_title('(b) MOOC → LastFM', loc='left', pad=40)
    ax.legend(title='Pretraining seed', title_fontsize=8, frameon=False, ncol=3,
              loc='lower center', bbox_to_anchor=(.5, 1.01), columnspacing=1.2)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='.9', linewidth=.6, zorder=0)
        ax.tick_params(width=.65, length=3)
    out = ROOT/'plots/outputs'
    out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'tfra_seed_interaction.{ext}', dpi=220)
    plt.close(fig)

if __name__ == '__main__':
    main()
