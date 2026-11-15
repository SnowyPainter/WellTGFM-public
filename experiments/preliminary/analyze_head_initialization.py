"""Audit all preregistered Head interventions and report complete crossed matrices."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from experiments.target_adaptation.runtime import ROOT,load_inputs,digest,array_hash
from experiments.common.runtime import atomic_json

OUT=Path(__file__).resolve().parent/'head_initialization_v1'

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    plan=json.loads((OUT/'plan.json').read_text());assert len(plan['jobs'])==72
    _,cfgs,data=load_inputs();rows=[];heads={};evaluations={};checks=[]
    for job in plan['jobs']:
        p=OUT/job['id'];r=json.loads((p/'result.json').read_text());m=json.loads((p/'manifest.json').read_text());base=job['base'];target=base['target']
        assert r['status']=='complete' and r['job']==m['job']==job and not r['test_used']
        assert m['runner_sha256']==digest(OUT.parent/'head_initialization.py')
        assert m['config']==cfgs[target]
        assert r['intervention']['non_head_state_identical'] and r['intervention']['postbuild_rng_identical']
        key=(target,job['head_seed']);h=r['intervention']['head_sha256'];assert heads.setdefault(key,h)==h
        if base['sources']:assert digest(ROOT/base['checkpoint'])==base['checkpoint_sha256']
        d=data[target];ids=d.indices(1)
        with np.load(p/'validation_predictions.npz') as a:
            assert np.array_equal(a['event_ids'],d.edge_id[ids])
            assert np.array_equal(a['negatives'],d.negatives(100007)[ids])
            assert np.array_equal(a['labels'],np.tile([1,0],(len(ids),1)))
            eh=array_hash(a);assert evaluations.setdefault(target,eh)==eh==r['evaluation_sha256']
            assert np.isfinite(a['scores']).all()
            ap=average_precision_score(a['labels'].ravel(),a['scores'].ravel());assert abs(ap-r['ap'])<1e-10
        hist=r['history'];assert [h['epoch'] for h in hist]==list(range(1,6))
        aps=[h['validation']['ap'] for h in hist];assert max(aps)==r['ap'] and 1+int(np.argmax(aps))==r['best_epoch'] and aps[-1]==r['last_epoch_ap']
        reused='reused_from' in r
        assert reused==(base['adaptation_seed']==job['head_seed'])
        if reused:
            ref=ROOT/r['reused_from'];assert digest(ref/'result.json')==r['reused_result_sha256']
            assert r['diagonal_identity_checked']
            assert digest(p/'validation_predictions.npz')==digest(ref/'validation_predictions.npz')
        rows.append(dict(job_id=job['id'],case_id=base['case_id'],target=target,source=bool(base['sources']),adaptation_seed=base['adaptation_seed'],head_seed=job['head_seed'],ap=r['ap'],last_epoch_ap=r['last_epoch_ap'],best_epoch=r['best_epoch'],reused=reused))
        checks.append(dict(job_id=job['id'],ap=float(ap),evaluation_sha256=eh,reused=reused))
    factors=[];matrices={};seeds=plan['head_seeds']
    for case in sorted({r['case_id'] for r in rows}):
        rs=[r for r in rows if r['case_id']==case];assert len(rs)==9
        for metric in ['ap','last_epoch_ap']:
            A=np.array([[next(r[metric] for r in rs if r['adaptation_seed']==a and r['head_seed']==h) for h in seeds] for a in seeds])
            mu=A.mean();ar=A.mean(1)-mu;hr=A.mean(0)-mu;inter=A-mu-ar[:,None]-hr[None,:]
            delta=A-np.diag(A)[:,None];off=delta[~np.eye(3,dtype=bool)]
            factors.append(dict(case_id=case,target=rs[0]['target'],source=rs[0]['source'],metric=metric,mean_ap=float(mu),head_rms=float(np.sqrt(np.mean(hr**2))),other_adaptation_rms=float(np.sqrt(np.mean(ar**2))),interaction_rms=float(np.sqrt(np.mean(inter**2))),mean_head_range=float(np.ptp(A,axis=1).mean()),max_head_range=float(np.ptp(A,axis=1).max()),improved=int(sum(off>0)),declined=int(sum(off<0)),max_improvement=float(off.max()),max_decline=float(off.min())))
            matrices[case,metric]=A
    write_csv(OUT/'comparison.csv',rows);write_csv(OUT/'factor_decomposition.csv',factors)
    atomic_json(OUT/'verification.json',dict(status='passed',checked_runs=len(rows),new_runs=sum(not r['reused'] for r in rows),reused_runs=sum(r['reused'] for r in rows),same_head_across_adaptation_seeds=True,test_used=False,records=checks))
    lines=['# Target Head 초기화와 다른 적응 난수의 교차 대조','','사전 고정한 네 Source 구성과 네 Target-only, 적응 seed 7·17·37 × Head seed 7·17·37의 72조건을 모두 평가했다. 기존 대각선24조건을 재사용하고48조건을 새로 학습했다. Source 사전학습 seed는7이며 Test는 사용하지 않았다.','','Head 외 모델 상태와 모델 생성 후 난수 상태의 동일성을 실행 중 확인했다. 같은 Target·Head seed의 초기 Head는 모든 적응 seed와 Source/Target-only 조건에서 동일했다. 아래는 관측한3개 seed의 기술통계다.','','## 효과 분해','','RMS와 범위는 AP 단위다. 행은 다른 적응 난수, 열은 Head seed다. Head 범위는 같은 적응 seed 안에서 Head만 바꿨을 때 AP의 최대−최소다. 상승/하락은 대각선 원래 Head 대비6개 교체의 방향이다.','','| 구성 | 평가 | Head RMS | 다른 난수 RMS | 상호작용 RMS | 평균 Head 범위 | 최대 Head 범위 | 상승/하락 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in factors:
        lines.append(f"| {r['case_id']} | {r['metric']} | "+' | '.join(f"{r[k]:.6f}" for k in ['head_rms','other_adaptation_rms','interaction_rms','mean_head_range','max_head_range'])+f" | {r['improved']}/{r['declined']} |")
    lines+=['','## 전체 행렬','','각 행은 다른 적응 난수 seed이며 세 열은 Head seed다. 최고 Validation epoch와 마지막 epoch를 모두 보고한다.']
    for (case,metric),A in matrices.items():
        lines+=['',f'### {case} — {metric}','','| 적응 seed | Head 7 | Head 17 | Head 37 |','|---|---:|---:|---:|']
        for seed,row in zip(seeds,A):lines.append(f'| {seed} | '+' | '.join(f'{x:.6f}' for x in row)+' |')
    lines+=['','## 해석 범위','','Source 구성은 AP를 보지 않고 Target별 Source 수 최소·동률 이름순으로 정했다. LastFM←MOOC는 기존 관측과 중복되는 조건이다. 기존 개발 패널에서 고른 구성으로 독립 Test나 새로운 Target은 아니다. 최고 Head의 결과를 방법 성능으로 선택하지 않으며, Head 초기화의 영향과 특정 적응 방법의 효과를 구분한다.','','[사전 설계](DESIGN.md) · [조건별 결과](comparison.csv) · [효과 분해](factor_decomposition.csv) · [검산](verification.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(factors,indent=2))

if __name__=='__main__':main()
