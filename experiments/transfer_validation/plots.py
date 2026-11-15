"""Export standalone diagnostic figures from completed, explicitly labeled results."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def save(fig,out,name):
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def render(out):
    root=out/'analysis';dest=out/'figures';dest.mkdir(parents=True,exist_ok=True)
    curves=[r for r in read(root/'learning_curves.csv') if r['cohort']=='long_budget' and r['sources']]
    targets=sorted({r['target'] for r in curves})
    if targets:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,t in zip(axes.ravel(),targets):
            for method in sorted({r['method'] for r in curves}):
                g=defaultdict(list)
                for r in curves:
                    if r['target']==t and r['method']==method:g[int(r['epoch'])].append(float(r['ap']))
                x=sorted(g);ax.plot(x,[np.mean(g[e]) for e in x],label=method)
            ax.set(title=t,xlabel='Epoch',ylabel='Validation AP')
            ax.grid(alpha=.2)
        axes.ravel()[0].legend(fontsize=8)
        save(fig,dest,'long_budget_validation_curves')
    rows=[r for r in read(root/'test_summary.csv') if r['cohort']=='query_budget' and r['evaluation']=='random_1' and r['target']!='target_equal_mean']
    if rows:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,t in zip(axes.ravel(),sorted({r['target'] for r in rows})):
            for method in sorted({r['method'] for r in rows}):
                g=sorted([r for r in rows if r['target']==t and r['method']==method],key=lambda r:float(r['query_fraction']))
                ax.plot([100*float(r['query_fraction']) for r in g],[float(r['ap']) for r in g],marker='o',label=method)
            ax.set(title=t,xlabel='Supervised Train queries (%)',ylabel='Test AP');ax.grid(alpha=.2)
        axes.ravel()[0].legend(fontsize=8)
        save(fig,dest,'supervised_query_budget_test')
    rows=read(root/'gradient_diagnostics.csv')
    if rows:
        fig,ax=plt.subplots(figsize=(7,5))
        for method in sorted({r['method'] for r in rows}):
            g=[r for r in rows if r['method']==method and float(r['epsilon'])>0 and float(r['gradient_gap'])>0]
            ax.scatter([float(r['epsilon']) for r in g],[float(r['gradient_gap']) for r in g],s=12,alpha=.5,label=method)
        ax.set(xscale='log',yscale='log',xlabel='Full Train readout fit error',ylabel='Full Train encoder gradient discrepancy')
        ax.legend();ax.grid(alpha=.2)
        save(fig,dest,'measured_readout_gradient_discrepancy')
