"""Completed three-seed budget evaluation; values from the frozen run summary."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/robust/outputs/v2_tgn/analysis/budget_summary.csv'
STYLES = [
    ('finetune','Fine-tuning','#555555','o','--'),
    ('lp_ft','LP-FT','#B16B36','s','--'),
    ('refine','REFINE','#8A7B9E','^','--'),
    ('joint','Joint MSE','#8C8650','v',':'),
    ('tfra','TFRA-E','#39658C','D','-'),
    ('batch','TFRA-B','#B44957','P','-'),
    ('initial_fixed','TFRA-Fixed','#308773','<','-'),
    ('joint_init','TFRA-Joint','#42643C','>','-'),
]


def main():
    with SOURCE.open() as f:
        rows = [r for r in csv.DictReader(f) if r['pretrain_seed']=='all']
    data = {(r['method'],json.loads(r['settings'])['budget']):r for r in rows}
    assert len(data)==24 and all(int(r['transfer_runs'])==42 for r in rows)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
                         'font.size':9,'axes.titlesize':10,'legend.fontsize':8,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65})
    fig, axes = plt.subplots(2,2,figsize=(9,6.8))
    axs = axes.ravel()
    fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.93,wspace=.34,hspace=.55)
    budgets = [5,20,50]
    for ax,key,title,label in zip(axs[:3],['D_G','B_G','AP'],
            ['(a) Negative transfer damage ↓','(b) Positive transfer benefit ↑','(c) Prediction performance ↑'],
            [r'$D_G$ (AP percentage points)',r'$B_G$ (AP percentage points)','Test AP (%)']):
        for method,name,color,marker,ls in STYLES:
            y = [100*float(data[method,b][key]) for b in budgets]
            ax.plot(budgets,y,label=name,color=color,marker=marker,linestyle=ls,
                    lw=1.35,ms=4.5,markeredgecolor='white',markeredgewidth=.4)
        ax.set(xlabel='Adaptation epochs',xticks=budgets,xlim=(2,53),ylabel=label)
        ax.set_title(title,loc='left',pad=11)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='.9',lw=.6)
        ax.tick_params(width=.65,length=3)
    axs[0].set(ylim=(0,1.6),yticks=[0,.4,.8,1.2,1.6])
    axs[1].set(ylim=(0,2.6),yticks=[0,.5,1,1.5,2,2.5])
    axs[2].set(ylim=(70,81),yticks=[70,72,74,76,78,80])
    timing_path = ROOT/'experiments/execution/outputs/efficiency_v1/summary.json'
    timing = {r['method']:r for r in json.loads(timing_path.read_text())}
    assert set(timing)=={s[0] for s in STYLES} and all(r['runs']==12 for r in timing.values())
    ax = axs[3]
    for i,(method,name,color,marker,ls) in enumerate(STYLES):
        seconds=timing[method]['seconds']
        ax.barh(i,seconds,height=.62,color=color,alpha=.85)
        ax.text(seconds+1,i,f'{seconds:.2f}',va='center',fontsize=8)
    ax.set(yticks=range(len(STYLES)),yticklabels=[s[1] for s in STYLES],
           xlabel='Mean wall-clock time (seconds)',xlim=(0,78))
    ax.invert_yaxis()
    ax.set_title('(d) Runtime at 5 epochs ↓',loc='left',pad=20)
    ax.text(0,1.025,'4 configurations × 3 seeds per method',transform=ax.transAxes,fontsize=8,color='.35')
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x',color='.9',lw=.6)
    ax.tick_params(width=.65,length=3)
    handles,labels=axs[0].get_legend_handles_labels()
    # Matplotlib fills legends by columns: arrange two semantic rows.
    order=[0,4,1,5,2,6,3,7]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],ncol=4,
               loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,
               columnspacing=2,handlelength=2.6)
    out=ROOT/'plots/outputs';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        destination = out/f'tfra_adaptation_budget.{ext}'
        fig.savefig(destination,dpi=220)
        if ext == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__=='__main__':
    main()
