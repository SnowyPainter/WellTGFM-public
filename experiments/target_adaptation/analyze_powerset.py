"""Descriptive analysis of the completed fixed-checkpoint powerset evaluation."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'outputs/powerset_v1'
M=['finetune','ridge_epoch'];LABEL={'finetune':'Fine-tuning','ridge_epoch':'TFRA'}

def markdown(frame):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def main():
    assert json.loads((OUT/'verification.json').read_text())['passed']
    plan=json.loads((OUT/'plan.json').read_text());jobs={j['job_id']:j for j in plan['jobs']}
    r=pd.read_csv(OUT/'comparison.csv');e=pd.read_csv(OUT/'source_addition.csv')
    assert len(r)==372 and len(e)==900
    r['delta_ap']=r.ridge_epoch_ap-r.finetune_ap
    r['delta_gain']=r.ridge_epoch_gain-r.finetune_gain
    r['development']=r.job_id.map(lambda x:'reuse_id' in jobs[x])
    e['seed']=e.before.map(lambda x:jobs[x]['adaptation_seed'])
    e['from_count']=e.before.map(lambda x:len(jobs[x]['sources']))
    e['before_case']=e.before.map(lambda x:jobs[x]['case_id'])
    e['after_case']=e.after.map(lambda x:jobs[x]['case_id'])
    idx=r.set_index('job_id')
    for m in M:
        e[m+'_before_gain']=e.before.map(idx[m+'_gain'])
        e[m+'_after_gain']=e.after.map(idx[m+'_gain'])
    cm=r.groupby(['target','case_id','source_count'],as_index=False)[[m+s for m in M for s in ['_ap','_gain']]+['delta_ap','delta_gain']].mean()
    em=e.groupby(['target','before_case','after_case','added','from_count'],as_index=False)[M].mean()
    lines=['# TFRA 전체 Source 조합: 상세 분석','',
        '고정된 사전학습 seed 7, 적응 seed 7·17·37, 네 Target의 비어 있지 않은 Source 구성 124개(전이 372회). 각 방법은 동일 절차의 Target-only와 비교한다. 기존 개발에 사용한 14구성을 포함하며, 아래 “확장 구성”은 이 14구성 밖의 110구성이다. 확장 구성도 기존 연구에서 관측한 Source/Target이며 독립 Test가 아니다. Test는 사용하지 않았다.',
        '', '900개 추가 비교는 동일 체크포인트와 평가 Query를 공유한다. 아래 빈도는 이 유한 평가 집합의 기술통계이며 독립 표본의 성공 확률이나 유의성 검정으로 해석하지 않는다. AP는 0–1 단위다.']
    def section(title,frame,note=None):
        lines.extend(['',f'## {title}','',markdown(frame)])
        if note:lines.extend(['',note])
    broad=[]
    for t,g in [('전체',r)]+list(r.groupby('target')):
        c=cm if t=='전체' else cm[cm.target==t]
        broad.append(dict(Target=t,N=len(g),AP_상승=int((g.delta_ap>0).sum()),AP_005초과상승=int((g.delta_ap>.005).sum()),AP_차이평균=g.delta_ap.mean(),AP_차이중앙값=g.delta_ap.median(),AP_차이최소=g.delta_ap.min(),AP_차이최대=g.delta_ap.max(),구성평균상승=int((c.delta_ap>0).sum()),구성수=len(c),세seed모두상승=int(g.groupby('case_id').delta_ap.min().gt(0).sum())))
    section('1. 절대 AP 개선의 범위',pd.DataFrame(broad))
    transfer=[]
    for t,g in [('전체',r)]+list(r.groupby('target')):
        for m in M:
            v=g[m+'_gain']
            transfer.append(dict(Target=t,방법=LABEL[m],양의전이=int((v>0).sum()),음의전이=int((v<0).sum()),N=len(v),평균G=v.mean(),DG=(-v).clip(lower=0).mean(),BG=v.clip(lower=0).mean(),최악G=v.min(),G하위10분위=v.quantile(.1),음수005=int((v<-.005).sum()),세seed모두양수=int(g.groupby('case_id')[m+'_gain'].min().gt(0).sum())))
    section('2. Target-only 대비 전이',pd.DataFrame(transfer),'각 방법의 Target-only가 다르므로 절대 AP 차이와 전이 이득 차이를 구분한다.')
    transitions=[]
    for t,g in [('전체',r)]+list(r.groupby('target')):
        transitions.append(dict(Target=t,양수유지=int(((g.finetune_gain>0)&(g.ridge_epoch_gain>0)).sum()),양수상실=int(((g.finetune_gain>0)&(g.ridge_epoch_gain<=0)).sum()),음수에서양수=int(((g.finetune_gain<0)&(g.ridge_epoch_gain>0)).sum()),음수잔존=int(((g.finetune_gain<0)&(g.ridge_epoch_gain<0)).sum()),전이이득상승=int((g.delta_gain>0).sum())))
    section('3. 양의 전이 보존과 음의 전이 회복',pd.DataFrame(transitions))
    controls=[]
    for (t,a),g in r.groupby(['target','adaptation_seed']):
        controls.append(dict(target=t,seed=a,**{m:float((g[m+'_ap']-g[m+'_gain']).iloc[0]) for m in M}))
    controls=pd.DataFrame(controls)
    decomposition=[]
    for t,g in r.groupby('target'):
        ct=controls[controls.target==t]
        decomposition.append(dict(Target=t,FT_Target_only=ct.finetune.mean(),TFRA_Target_only=ct.ridge_epoch.mean(),Target_only차이=ct.ridge_epoch.mean()-ct.finetune.mean(),전이AP차이=g.delta_ap.mean(),전이G차이=g.delta_gain.mean()))
    section('4. AP 향상과 사전학습 활용 향상의 구분',pd.DataFrame(decomposition),'전이 AP 차이 = Target-only AP 차이 + 전이 이득 G 차이. Target-only가 낮아지면 G는 증가할 수 있으므로 세 항을 함께 해석한다.')
    common=[]
    for t,g in [('전체',r)]+list(r.groupby('target')):
        v=g.ridge_epoch_ap-(g.finetune_ap-g.finetune_gain)
        common.append(dict(Target=t,N=len(g),FT_Target_only초과=int(v.gt(0).sum()),평균차이=v.mean(),평균하락폭=(-v).clip(lower=0).mean(),하락005=int(v.lt(-.005).sum())))
    section('4.1. 공통 Fine-tuning Target-only를 기준으로 한 보조 비교',pd.DataFrame(common),'이 표는 동일 절차 Target-only 기반 G를 대체하지 않는다. 서로 다른 적응 절차의 scratch 성능 차이가 전이 해석에 미치는 영향을 확인하는 보조 비교다.')
    edge_rows=[]
    for name,g in [('전체',e)]+[(t,g) for t,g in e.groupby('target')]+[(f'{k}→{k+1}',g) for k,g in e.groupby('from_count')]:
        for m in M:
            v=g[m];edge_rows.append(dict(조건=name,방법=LABEL[m],상승=int((v>0).sum()),N=len(v),상승비율=(v>0).mean(),평균C=v.mean(),중앙값C=v.median(),DC=(-v).clip(lower=0).mean(),BC=v.clip(lower=0).mean(),상승005=int((v>.005).sum()),하락005=int((v<-.005).sum())))
    section('5. Source 하나 추가: 대개 좋아지는가?',pd.DataFrame(edge_rows),'단일 추가의 C>0 빈도가 “대개 추가하면 좋아진다”를 직접 평가하는 지표다. 평균 AP가 높다는 사실과 추가 기여가 양수라는 사실은 다르다.')
    seed_edges=[]
    for t,g in [('전체',e)]+list(e.groupby('target')):
        for m in M:
            z=g.groupby(['before_case','after_case'])[m]
            seed_edges.append(dict(Target=t,방법=LABEL[m],경로수=z.ngroups,평균C양수=int(z.mean().gt(0).sum()),세seed모두상승=int(z.min().gt(0).sum()),세seed모두하락=int(z.max().lt(0).sum()),seed간부호반전=int(((z.min()<0)&(z.max()>0)).sum())))
    section('6. Source 추가 효과의 seed 일관성',pd.DataFrame(seed_edges))
    strat=[]
    for (t,k),g in e.groupby(['target','from_count']):
        for m in M:strat.append(dict(Target=t,추가=f'{k}→{k+1}',방법=LABEL[m],상승=int(g[m].gt(0).sum()),N=len(g),평균C=g[m].mean()))
    section('7. Target × Source 수별 추가 효과',pd.DataFrame(strat))
    source=[]
    for (t,added),g in e.groupby(['target','added']):
        for m in M:source.append(dict(Target=t,추가Source=added,방법=LABEL[m],상승=int(g[m].gt(0).sum()),N=len(g),평균C=g[m].mean(),DC=(-g[m]).clip(lower=0).mean()))
    section('8. 어떤 Source가 추가될 때 도움이 되는가?',pd.DataFrame(source),'각 Source의 효과는 모든 가능한 기존 구성에 걸친 요약이다. 이 결과를 보아 Source를 고르는 것은 사후 선택이며, 미관측 Target에 적용할 사전 선택 규칙은 아니다.')
    # Existence of beneficial extensions, based on 3-seed means, not an operational selector.
    opportunities=[]
    for t,g in [('전체',em)]+list(em.groupby('target')):
        for m in M:
            z=g.groupby('before_case')[m]
            opportunities.append(dict(Target=t,방법=LABEL[m],확장가능구성=z.ngroups,개선추가존재=int(z.max().gt(0).sum()),모든추가개선=int(z.min().gt(0).sum()),어떤추가도개선없음=int(z.max().le(0).sum())))
    section('9. 적어도 하나의 유익한 Source 추가가 존재하는가?',pd.DataFrame(opportunities),'세 seed 평균에서 Source가 1–4개인 기존 구성당 가능한 추가를 조사한다. “개선 가능한 추가가 존재”와 “임의로 추가해도 대개 개선”은 서로 다른 주장이다. 최대값은 가능성의 사후 기술통계이며 선택 알고리즘 성과가 아니다.')
    counts=[]
    for (t,k),g in r.groupby(['target','source_count']):
        for m in M:counts.append(dict(Target=t,Source수=k,방법=LABEL[m],AP=g[m+'_ap'].mean(),G=g[m+'_gain'].mean(),DG=(-g[m+'_gain']).clip(lower=0).mean(),BG=g[m+'_gain'].clip(lower=0).mean()))
    section('10. Source 수별 절대 성능과 전이 이득',pd.DataFrame(counts))
    # All 120 source permutations per target/seed, share endpoints. Count paths as descriptive only.
    path_rows=[]
    for t,cfg in plan['configs'].items():
        pool=[x for x in cfg['datasets'] if x!=t]
        for m in M:
            look={(frozenset(jobs[x]['sources']),int(a)):v for x,a,v in zip(r[r.target==t].job_id,r[r.target==t].adaptation_seed,r[r.target==t][m+'_ap'])}
            paths=[]
            for seed in [7,17,37]:
                for order in itertools.permutations(pool):
                    v=np.array([look[frozenset(order[:k]),seed] for k in range(1,6)])
                    paths.append(dict(increases=int((np.diff(v)>0).sum()),monotonic=bool((np.diff(v)>0).all()),end_gain=float(v[-1]-v[0])))
            path_rows.append(dict(Target=t,방법=LABEL[m],경로수=len(paths),전단계상승=sum(x['monotonic'] for x in paths),네번중평균상승수=mean([x['increases'] for x in paths]),끝점5개가1개보다상승비율=mean([x['end_gain']>0 for x in paths]),끝점차이평균=mean([x['end_gain'] for x in paths])))
    section('11. 1→2→3→4→5 순차 확장 경로',pd.DataFrame(path_rows),'각 Target에서 5 Source의 120개 순열 × 3 seeds를 열거했다. 구성마다 독립 사전학습한 모델의 비교이며 순차 학습 실험은 아니다. 경로들은 같은 구성과 끝점을 반복 사용한다.')
    # Stability: paired per-composition 3 seeds.
    stability=[]
    for (t,case),g in r.groupby(['target','case_id']):
        assert len(g)==3
        stability.append(dict(target=t,case_id=case,**{m+'_sd':g[m+'_ap'].std(ddof=0) for m in M},**{m+'_min':g[m+'_ap'].min() for m in M}))
    st=pd.DataFrame(stability);sr=[]
    for t,g in [('전체',st)]+list(st.groupby('target')):
        sr.append(dict(Target=t,N=len(g),SD감소=int((g.ridge_epoch_sd<g.finetune_sd).sum()),FT평균SD=g.finetune_sd.mean(),TFRA평균SD=g.ridge_epoch_sd.mean(),최악seedAP개선=int((g.ridge_epoch_min>g.finetune_min).sum()),최악seedAP차이평균=(g.ridge_epoch_min-g.finetune_min).mean()))
    section('12. 구성별 seed 안정성과 최악 seed 성능',pd.DataFrame(sr))
    slices=[]
    for name,g in [('개발14구성',r[r.development]),('확장110구성',r[~r.development])]+[(f'adaptation seed {a}',g) for a,g in r.groupby('adaptation_seed')]:
        for m in M:slices.append(dict(구분=name,방법=LABEL[m],N=len(g),AP=g[m+'_ap'].mean(),DG=(-g[m+'_gain']).clip(lower=0).mean(),BG=g[m+'_gain'].clip(lower=0).mean(),AP비교승수=int(g.delta_ap.gt(0).sum())))
    section('13. 개발 구성 밖과 각 적응 seed에서의 성과',pd.DataFrame(slices),'개발/확장 구성의 Target 비중은 서로 다르므로 각 행은 해당 집합의 조건 평균이다. 같은 집합 안에서 방법을 비교한다. AP 비교 승수는 TFRA > Fine-tuning인 조건 수다.')
    failures=r.sort_values('delta_ap').head(12)
    section('14. AP 개선이 작은 조건과 악화 사례',failures[['target','case_id','adaptation_seed','finetune_ap','ridge_epoch_ap','delta_ap','finetune_gain','ridge_epoch_gain']])
    strongest=em.sort_values('ridge_epoch',ascending=False).head(10)
    section('15. 세 seed 평균 Source 추가 이득 상위 경로',strongest,'상위 사례는 사후 예시이며 전체 빈도는 §5–8을 기준으로 해석한다.')
    r.to_csv(OUT/'detailed_comparison.csv',index=False);e.to_csv(OUT/'detailed_source_addition.csv',index=False);st.to_csv(OUT/'stability_by_composition.csv',index=False)
    conclusions=['', '## 핵심 해석', '',
        '- 임의의 Source 하나 추가는 TFRA에서 412/900건(45.8%) 상승했다. 전체 평균 C도 음수이므로 “Source를 추가하면 대개 좋아진다”는 일반적 주장은 지지하지 않는다.',
        '- 반면 세 seed 평균에서 확장 가능한 120개 구성 중 73개(60.8%)에는 적어도 하나의 개선 추가가 있다. 이는 유익한 확장 기회의 존재이며 이를 찾는 선택 규칙의 검증은 아니다.',
        '- 1→2 추가는 130/240건(54.2%) 상승하고 평균 +0.003877 AP, 4→5는 33/60건(55.0%) 상승하나 평균 +0.000099 AP다. 2→3과 3→4는 각각 44.2%, 37.5% 상승이다.',
        '- TFRA는 동일 체크포인트 Fine-tuning보다 301/372건(80.9%) 높은 AP, 세 seed 평균으로 106/124구성(85.5%)의 개선을 보였다. 77/124구성은 세 seed 모두 개선됐다.',
        '- Fine-tuning 음수 전이 199건 중 157건(78.9%)이 TFRA에서 양수로 전환되고, 기존 양수 173건 중 161건(93.1%)이 유지됐다. 각 방법의 자체 Target-only 기준이다.',
        '- 세 seed 모두 양의 전이인 구성은 Fine-tuning 19/124에서 TFRA 90/124로 증가했다. 동일 방법 Target-only 기준이며 사전학습 seed 7에 한정된다.',
        '- 최악 적응 seed의 AP가 112/124구성(90.3%)에서 높아졌고, 그 최솟값의 평균 개선은 +0.030347 AP다. 구성별 SD는 87/124구성에서 감소했다.',
        '- 기존 개발 밖 110구성의 330회에서도 AP가 267회(80.9%) 개선됐고, 평균 +0.024815 AP였다. 세 적응 seed를 각각 집계해도 AP·DG·BG가 모두 개선됐다.',
        '- LastFM은 절대 전이 AP가 +0.004983 개선됐지만 TFRA Target-only가 Fine-tuning Target-only보다 0.021890 낮다. 전이 G 증가 +0.026873 중 큰 부분은 이 기준 차이와 연결된다. 공통 Fine-tuning Target-only를 넘는 TFRA LastFM 실행은 40/93회다.',
        '- Wikipedia·Reddit·MOOC는 TFRA Target-only 자체도 개선되면서, 사전학습 전이 AP와 G가 함께 개선됐다. 이 세 Target에서는 단순히 낮은 scratch 기준으로 G가 커졌다는 설명이 성립하지 않는다.',
        '- Source 수 5개에서는 평균 seed SD가 Fine-tuning보다 증가한다(기존 전체 결과 표 참고). 평균 안정성 개선을 모든 구성의 안정성 보장으로 해석하지 않는다.',
        '', '모든 분석은 기존 사전학습 seed 7의 동일 모델과 Validation에 한정된다. 구성당 세 seed의 최솟값은 관측된 세 실행의 최솟값이며 모집단 최악 성능 보장은 아니다.']
    lines[4:4]=conclusions+['', '![Source 수별 전이 AP](../../../../plots/outputs/tfra_source_count.png)', '', '실선은 각 Target·Source 수의 모든 구성과 세 적응 seed의 평균 AP, 점선은 각 방법 Target-only의 세 seed 평균이다. Source당 학습량이 아닌 총 사전학습 업데이트 60회가 고정된 비교다.', '']
    report='\n'.join(lines)+'\n';(OUT/'DETAILED_ANALYSIS.md').write_text(report)
    for title,data in [('broad',broad),('edges',edge_rows),('opportunities',opportunities),('stability',sr),('slices',slices),('paths',path_rows),('transitions',transitions),('decomposition',decomposition)]:
        print(title);print(pd.DataFrame(data).to_string(index=False))

from statistics import mean
if __name__=='__main__':main()
