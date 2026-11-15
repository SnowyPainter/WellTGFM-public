"""Prior evaluation scopes and the controlled evidence motivating this study.

Literature scope: docs/선행연구분석.md, §7 (local PDF audit).
Empirical values: data/tfra_problem_overview.json; our TGAT experiments.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, RED = '#39658C', '#A55342'


def main():
    data = json.loads((HERE / 'data/tfra_problem_overview.json').read_text())
    edges = data['source_improvement_paths']
    assert len(edges) == 219
    assert sum(e['target_delta'] < 0 for e in edges) == 98
    head = data['head_intervention']
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'font.size': 9,
                         'axes.linewidth': .65, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(8.4, 4.5))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.set(xlim=(0, 1), ylim=(0, 1)); canvas.axis('off')

    def txt(x, y, s, **kw):
        canvas.text(x, y, s, va='top', **kw)

    txt(.02, .985, 'PRIOR EVALUATION', fontsize=8, color='.35')
    txt(.05, .935, 'DyGFM', weight='bold', fontsize=11)
    txt(.05, .88, 'Disentanglement + routing / prompting\nImproved downstream prediction', linespacing=1.4)
    txt(.55, .935, 'TIDES', weight='bold', fontsize=11)
    txt(.55, .88, 'Disentangled temporal / structural priors\nImproved zero-shot transfer', linespacing=1.4)
    txt(.05, .77, 'Without a matched Target-only comparison:\nhow much does pretraining add?', color=BLUE, linespacing=1.35)
    txt(.55, .77, 'Beyond zero-shot evaluation:\nwhat changes during Target adaptation?', color=BLUE, linespacing=1.35)
    for x in [.27, .76]:
        canvas.annotate('', xy=(x, .64), xytext=(x, .675),
                        arrowprops={'arrowstyle': '->', 'lw': .8, 'color': '.4'})
    canvas.plot([.02, .98], [.625, .625], color='.65', lw=.6)
    txt(.02, .604, 'OUR CONTROLLED STUDY  ·  TGAT / temporal link prediction', fontsize=8, color='.35')
    txt(.05, .552, '(a) Source improvement is insufficient', weight='bold', fontsize=9.5)
    txt(.55, .552, '(b) Checkpoint utility depends on adaptation', weight='bold', fontsize=9.1)

    ax = fig.add_axes([.09, .215, .31, .26])
    x = 100 * np.array([r['source_mean_delta'] for r in edges])
    y = 100 * np.array([r['target_delta'] for r in edges])
    for mask, color, marker in [(y >= 0, BLUE, 'o'), (y < 0, RED, 'v')]:
        ax.scatter(x[mask], y[mask], s=10, color=color, marker=marker, alpha=.65, linewidths=0)
    ax.axhline(0, color='.4', lw=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlim=(0, 5.6), ylim=(-6.3, 5), xticks=[0, 2, 4], yticks=[-6, -3, 0, 3])
    ax.set_xlabel('Mean Source AP change (pp)', fontsize=8, labelpad=2)
    ax.set_ylabel('Target AP change (pp)', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, length=2)
    txt(.05, .505, 'All retained Sources improve; Target falls in 98 / 219 paths.', fontsize=8)

    # A single checkpoint branches only through Head initialization.
    txt(.755, .505, 'MOOC → LastFM; all other adaptation RNG fixed', ha='center', fontsize=8)
    txt(.755, .45, 'Same pretrained checkpoint', ha='center', fontsize=9)
    vals = {r['head_seed']: r['ap'] for r in head}
    for x, seed in zip([.60, .755, .91], [7, 17, 37]):
        canvas.annotate('', xy=(x, .335), xytext=(.755, .405),
                        arrowprops={'arrowstyle': '->', 'lw': .8, 'color': '.4'})
        txt(x, .32, f'Head seed {seed}', ha='center', fontsize=8)
        txt(x, .265, f'{vals[seed]:.3f}', ha='center', fontsize=12, color=RED if seed == 7 else BLUE)
    txt(.755, .205, 'Validation AP', ha='center', fontsize=8)
    txt(.755, .16, 'Seed-crossing study: rank reversals in 31 / 32 configurations.', ha='center', fontsize=8)

    canvas.plot([.02, .98], [.115, .115], color='.65', lw=.6)
    txt(.5, .09, 'Measure the realized contribution of pretraining', weight='bold', ha='center', fontsize=9.5)
    txt(.5, .043, r'$G = \mathrm{AP}_{\mathrm{pretrained}} - \mathrm{AP}_{\mathrm{Target\!\!-\!\!only}}$'
        '   |   same adaptation rule, budget and seed', ha='center', fontsize=9)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(HERE / f'outputs/tfra_problem_overview.{ext}', dpi=240,
                    bbox_inches='tight', pad_inches=.04, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
