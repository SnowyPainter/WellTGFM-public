"""Summarize the final adaptation method against ordinary Fine-tuning."""
import csv,json
from statistics import mean,pstdev
from experiments.target_adaptation.runtime import HERE,load_inputs
from experiments.common.runtime import atomic_json

def main():
    jobs,_,_=load_inputs(False)
    results={(m,j['job_id']):json.loads((HERE/'outputs'/m/j['job_id']/'result.json').read_text()) for m in ['finetune','ridge_epoch'] for j in jobs}
    rows=[]
    for j in jobs:
        if not j['sources']:continue
        row={k:j[k] for k in ['job_id','case_id','target','adaptation_seed']}
        ctl=f"{j['target']}_target_only__pnone__a{j['adaptation_seed']}"
        for m in ['finetune','ridge_epoch']:
            row[m+'_ap']=results[m,j['job_id']]['ap']
            row[m+'_gain']=row[m+'_ap']-results[m,ctl]['ap']
        rows.append(row)
    def stats(rs,m):
        targets=sorted({r['target'] for r in rs})
        macro=lambda fn:mean(mean(fn(r) for r in rs if r['target']==t) for t in targets)
        groups={}
        for r in rs:groups.setdefault((r['target'],r['case_id']),[]).append(r[m+'_ap'])
        return dict(AP=macro(lambda r:r[m+'_ap']),D_G=macro(lambda r:max(-r[m+'_gain'],0)),B_G=macro(lambda r:max(r[m+'_gain'],0)),seed_SD=mean(mean(pstdev(v) for (tt,_),v in groups.items() if tt==t) for t in targets),negative=sum(r[m+'_gain']<0 for r in rs),negative_over_005=sum(r[m+'_gain']<-.005 for r in rs))
    summary={m:stats(rows,m) for m in ['finetune','ridge_epoch']}
    by_target={t:{m:stats([r for r in rows if r['target']==t],m) for m in summary} for t in sorted({r['target'] for r in rows})}
    old=[r for r in rows if r['finetune_gain']>0]
    retention=dict(baseline_positive=len(old),retained=sum(r['ridge_epoch_gain']>0 for r in old),lost=sum(r['ridge_epoch_gain']<=0 for r in old))
    atomic_json(HERE/'outputs/summary.json',dict(overall=summary,by_target=by_target,retention=retention))
    with (HERE/'outputs/comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Target-Fitted Readout Adaptation (TFRA) 결과','','사전학습 seed 7, 적응 seed 7·17·37, 고정된 14개 Source 구성. 방법당 전이 42회와 동일 방법 Target-only 12회다. Validation 평가이며 Test는 사용하지 않았다. 평균은 Target별 동일 비중이고 SD는 구성별 세 AP의 분모 3 표준편차다.','','| 방법 | AP | D_G | B_G | Seed SD |','|---|---:|---:|---:|---:|']
    labels={'finetune':'Fine-tuning','ridge_epoch':'TFRA'}
    for m,s in summary.items():lines.append('| '+labels[m]+' | '+' | '.join(f'{s[k]:.6f}' for k in ['AP','D_G','B_G','seed_SD'])+' |')
    lines+=['','## Target별 결과','','| Target | 방법 | AP | D_G | B_G | Seed SD |','|---|---|---:|---:|---:|---:|']
    for t,methods in by_target.items():
        for m,s in methods.items():lines.append('| '+t+' | '+labels[m]+' | '+' | '.join(f'{s[k]:.6f}' for k in ['AP','D_G','B_G','seed_SD'])+' |')
    lines+=['',f"Fine-tuning의 양의 전이 {retention['baseline_positive']}조건 중 {retention['retained']}조건을 양수로 유지하고 {retention['lost']}조건은 양수를 잃었다. 평균 개선이 모든 조건의 보존을 뜻하지 않는다.",'','MOOC의 AP와 전이 이득은 낮아졌고 Wikipedia의 seed SD는 소폭 높아졌다. 이 결과는 고정된 사전학습 초기 조건의 개발 패널에 한정된다.','','## 계산 설정','','Encoder Adam 학습률 1e-4, ridge 계수 1e-3, Dropout 0, 시간 배치당 균등 간격 256 Query, 5 epoch·60 업데이트다. Epoch 시작과 종료에 전체 Train 예측층을 적합하여 총 10회 적합한다. Fine-tuning은 전체 배치와 Dropout 0.1을 사용하므로 같은 계산량의 비교가 아니다. 원래 결과와 manifest는 각 실행 폴더에 보존했다.','','[조건별 수치](comparison.csv) · [집계 JSON](summary.json)']
    (HERE/'outputs/RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
