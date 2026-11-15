"""CPU-only per-target and epoch-trajectory audit of the fixed REFINE panel."""
import csv,json
from pathlib import Path
from collections import Counter
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/target_adaptation/outputs/refine_v1'
TARGETS=['wikipedia','reddit','mooc','lastfm']

def main():
    assert json.loads((OUT/'verification.json').read_text())['passed']
    results=[json.loads(p.read_text()) for p in (OUT/'refine').glob('*/result.json')]
    assert len(results)==54
    summaries={r['method']:r for r in json.loads((OUT/'summary.json').read_text())['methods']}
    comparison=list(csv.DictReader((OUT/'comparison.csv').open()))
    audit=[];epochs=[]
    for t in TARGETS:
        for transfer in [True,False]:
            rs=[r for r in results if r['job']['target']==t and bool(r['job']['sources'])==transfer]
            assert all(len(r['history'])==5 for r in rs)
            hist=dict(sorted(Counter(r['best_epoch'] for r in rs).items()))
            row=dict(target=t,group='transfer' if transfer else 'target_only',n=len(rs),best_epoch_counts=hist,
                     selected_AP=mean(r['ap'] for r in rs),last_AP=mean(r['last_epoch_ap'] for r in rs),
                     rising_last_step=sum(r['history'][4]['validation']['ap']>r['history'][3]['validation']['ap'] for r in rs))
            audit.append(row)
            for e in range(5):
                epochs.append(dict(target=t,group=row['group'],epoch=e+1,AP=mean(r['history'][e]['validation']['ap'] for r in rs),train_BCE=mean(r['history'][e]['train_loss'] for r in rs),validation_BCE=mean(r['history'][e]['validation']['loss'] for r in rs)))
    with (OUT/'epoch_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(epochs[0]));w.writeheader();w.writerows(epochs)
    (OUT/'trajectory_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# REFINE: Target and training-budget audit','', 'Fixed 14-configuration panel, pretraining seed 7, adaptation seeds 7/17/37, Validation only. Each method uses its own matched Target-only. This analysis uses completed runs; no extra training or hyperparameter selection.','', '## Target-level comparison','', '| Target | FT AP | LP-FT AP | TFRA AP | REFINE AP | REFINE Target-only AP | TFRA wins / n |','|---|---:|---:|---:|---:|---:|---:|']
    for t in TARGETS:
        aps=[next(x['AP'] for x in summaries[m]['by_target'] if x['target']==t) for m in ['finetune','lp_ft','ridge_epoch','refine']]
        rr={r['job_id']:float(r['AP']) for r in comparison if r['target']==t and r['method']=='refine'}
        wins=sum(float(r['AP'])>rr[r['job_id']] for r in comparison if r['target']==t and r['method']=='ridge_epoch')
        to=next(r['selected_AP'] for r in audit if r['target']==t and r['group']=='target_only')
        lines.append('| '+t+' | '+' | '.join(f'{v:.6f}' for v in aps+[to])+f' | {wins}/{len(rr)} |')
    lines+=['','## Epoch-selection and late trajectory','','| Target | Group | n | Best-epoch counts (1 / 2 / 3 / 4 / 5) | Selected mean AP | Final mean AP | AP rises 4→5 |','|---|---|---:|---|---:|---:|---:|']
    for r in audit:
        counts=' / '.join(str(r['best_epoch_counts'].get(e,0)) for e in range(1,6))
        lines.append(f"| {r['target']} | {r['group']} | {r['n']} | {counts} | {r['selected_AP']:.6f} | {r['last_AP']:.6f} | {r['rising_last_step']}/{r['n']} |")
    lines+=['','## Findings and decision','',
      '- MOOC: all nine transfer runs select epoch 5, and all nine improve from epoch 4 to 5. Two of three matched Target-only runs also select epoch 5. Five epochs do not demonstrate convergence here; an extended-budget comparison is warranted before claiming a general performance ranking.',
      '- Wikipedia, Reddit, LastFM: all 33 transfer runs select epochs 1–3. Their mean AP drops after the peak through epoch 5. Their training AND Validation BCE continue falling on average. This is divergence between the BCE objective and ranking quality, not evidence of conventional rising-validation-loss overfitting. A later recovery remains possible and is not ruled out by five observations.',
      '- The three early-peaking Targets account for most of the observed TFRA advantage. MOOC is nearly tied in selected AP, so insufficient MOOC training cannot by itself explain the overall gap.',
      '- REFINE has lower mean transfer AP than its matched Target-only on every Target in this panel. Both branches and their classifier use the same fixed optimization budget across transfer and Target-only. This describes the implemented TGAT variant; it does not invalidate the paper\'s distribution-dependent rate guarantees.',
      '- Retain the completed five-epoch comparison as the common-budget result. Before powerset expansion, the useful next check is a separately reported longer-budget REFINE run with matched Target-only, preserving the optimizer/RNG state and selecting only by Validation. Do not replace the five-epoch result with a favorable later run or label REFINE converged without that check. No extended run was launched by this audit.','',
      '## Curves','', '[PDF](../../../../plots/outputs/refine_learning_curves.pdf). Curves are means of each epoch, not means of the per-run selected maxima. Shading is omitted for readability; exact epoch means are in epoch_curves.csv.','']
    (OUT/'TRAINING_AUDIT.md').write_text('\n'.join(lines))
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for col,t in enumerate(TARGETS):
        for group,color,ls in [('transfer','#087f69','-'),('target_only','#777777','--')]:
            rr=[r for r in epochs if r['target']==t and r['group']==group]
            axes[0,col].plot(range(1,6),[r['AP'] for r in rr],color=color,ls=ls,marker='o',ms=3,label=group.replace('_',' '))
            for key,style in [('train_BCE',ls),('validation_BCE',':' if group=='transfer' else '-.')]:
                axes[1,col].plot(range(1,6),[r[key] for r in rr],color=color,ls=style,label=group.replace('_',' ')+' '+key.replace('_',' '))
        axes[0,col].set_title(t.capitalize());axes[0,col].set_ylabel('Validation AP');axes[1,col].set_ylabel('BCE')
        for ax in axes[:,col]:ax.set_xticks(range(1,6));ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=6)
    dest=ROOT/'plots/outputs'
    for ext in ['pdf','png']:fig.savefig(dest/f'refine_learning_curves.{ext}',dpi=180)
    print((OUT/'TRAINING_AUDIT.md').read_text())
if __name__=='__main__':main()
