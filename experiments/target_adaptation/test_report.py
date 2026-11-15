"""Recompute manuscript comparison tables from all frozen Test predictions."""
from collections import defaultdict
import difflib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import average_precision_score

from experiments.common.runtime import atomic_json
from experiments.transfer_validation.core import ROOT, BASE, read, digest
from experiments.transfer_validation.report import records, summarize, csv_write

MAIN=['finetune','lp_ft','refine','tfra']
PANEL=['finetune','frozen','warmstart','batch','joint','tfra']
TARGETS=['wikipedia','reddit','mooc','lastfm']
LABEL={'finetune':'Fine-tuning','lp_ft':'LP-FT','refine':'REFINE','tfra':'TFRA',
       'frozen':'Frozen Encoder','warmstart':'Ridge Warm-start','batch':'Batch-wise Ridge Profile','joint':'Joint MSE'}
TARGET_LABEL={'wikipedia':'Wikipedia','reddit':'Reddit','mooc':'MOOC','lastfm':'LastFM','target_equal_mean':'전체'}


def paired(transfer, baseline):
    by={(r['target'],r['sources'],r['adaptation_seed'],r['method']):r for r in transfer}
    keys=sorted({(r['target'],r['sources'],r['adaptation_seed']) for r in transfer})
    pairs=[(by[k+(baseline,)],by[k+('tfra',)]) for k in keys]
    assert len(pairs)==372
    pos=[(a,b) for a,b in pairs if a['gain']>0];neg=[(a,b) for a,b in pairs if a['gain']<0]
    groups=defaultdict(list)
    for a,b in pairs:groups[a['target'],a['sources']].append((a,b))
    return dict(baseline=baseline,wins=sum(b['ap']>a['ap'] for a,b in pairs),
        ties=sum(b['ap']==a['ap'] for a,b in pairs),negative=len(neg),positive=len(pos),
        negative_to_positive=sum(b['gain']>0 for a,b in neg),positive_retained=sum(b['gain']>0 for a,b in pos),
        original_positive_gain=float(np.mean([a['gain'] for a,b in pos])) if pos else None,
        tfra_on_original_positive_gain=float(np.mean([b['gain'] for a,b in pos])) if pos else None,
        all_seed_ap_improved=sum(all(b['ap']>a['ap'] for a,b in xs) for xs in groups.values()),
        mean_ap_improved=int(sum(np.mean([b['ap'] for a,b in xs])>np.mean([a['ap'] for a,b in xs]) for xs in groups.values())),
        min_ap_improved=sum(min(b['ap'] for a,b in xs)>min(a['ap'] for a,b in xs) for xs in groups.values()),
        seed_sd_reduced=int(sum(np.std([b['ap'] for a,b in xs])<np.std([a['ap'] for a,b in xs]) for xs in groups.values())))


def additions(transfer):
    by={(r['method'],r['target'],r['adaptation_seed'],frozenset(r['sources'].split('+'))):r for r in transfer}
    edges=[]
    for (m,t,a,s),before in by.items():
        for x in ['wikipedia','reddit','mooc','lastfm','ICEWS14','GDELT']:
            if x==t or x in s:continue
            after=by[m,t,a,s|{x}]
            edges.append(dict(method=m,target=t,adaptation_seed=a,before=before['sources'],after=after['sources'],
                              source_count=len(s),added=x,contribution=after['ap']-before['ap']))
    result={}
    for m in MAIN:
        x=[e for e in edges if e['method']==m];assert len(x)==900
        c=np.array([e['contribution'] for e in x])
        by_count={str(k):dict(positive=sum(e['contribution']>0 for e in x if e['source_count']==k),
            total=sum(e['source_count']==k for e in x)) for k in range(1,5)}
        result[m]=dict(mean=float(c.mean()),D_C=float(np.maximum(-c,0).mean()),
            positive=int((c>0).sum()),negative=int((c<0).sum()),zero=int((c==0).sum()),by_count=by_count)
    return edges,result


def dominates(a,b):
    return (a['ap']>=b['ap'] and a['D_G']<=b['D_G'] and a['B_G']>=b['B_G'] and
            (a['ap']>b['ap'] or a['D_G']<b['D_G'] or a['B_G']>b['B_G']))


def table8(summary):
    lines=['| Target | 방법 | Test AP ↑ | 전이 피해 \\(D_G\\) ↓ | 전이 이득 \\(B_G\\) ↑ | Seed SD ↓ | 큰 음수 전이 ↓ | 안정적 양의 전이 ↑ |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in TARGETS+['target_equal_mean']:
        group=[summary[m,t] for m in MAIN]
        for m in MAIN:
            r=summary[m,t];cells=[]
            for key,fmt,maximize in [('ap','.4f',True),('D_G','.6f',False),('B_G','.6f',True),('seed_sd','.6f',False)]:
                best=(max if maximize else min)(v[key] for v in group)
                s=format(r[key],fmt);cells.append('**'+s+'**' if r[key]==best else s)
            for key,den,maximize in [('large_negative',r['runs'],False),('stable_positive',r['stability_groups'],True)]:
                best=(max if maximize else min)(v[key] for v in group)
                s=f'{r[key]}/{den}';cells.append('**'+s+'**' if r[key]==best else s)
            lines.append('| '+TARGET_LABEL[t]+' | '+LABEL[m]+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def table9(panel):
    lines=['| 방법 | Target-only Test AP | 전이 Test AP ↑ | 평균 전이 이득 \\(\\bar G\\) ↑ | 전이 피해 \\(D_G\\) ↓ | 전이 이득 \\(B_G\\) ↑ |',
           '|---|---:|---:|---:|---:|---:|']
    for m in PANEL:
        r=panel[m]
        cells=[f"{r[k]:.6f}" for k in ['target_only_ap','ap']]+[f"{r['mean_gain']:+.6f}"]+[f"{r[k]:.6f}" for k in ['D_G','B_G']]
        lines.append('| '+LABEL[m]+' | '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def change(new,old):
    if old==0:return f'{old:.6f}에서 {new:.6f}로 변했다'
    pct=100*(new-old)/abs(old)
    return f'{abs(pct):.1f}% '+('증가했다' if pct>0 else '감소했다' if pct<0 else '로 유지됐다')


def manuscript(out, summary, panel, pairs, composition, count_sd):
    overall={m:summary[m,'target_equal_mean'] for m in MAIN};tf=overall['tfra'];ft=overall['finetune'];p=pairs['finetune']
    section53='''### 5.3. Transfer Performance on the Held-out Test Split

Validation으로 선택해 저장한 Fine-tuning, LP-FT, REFINE과 TFRA 모델을 전체 Source 조합의 Test에서 평가한다. 각 모델의 epoch와 설정을 확정한 뒤, 동일한 Test 후보에서 AP와 전이 피해·이득, 적응 seed 변동을 계산했다. Target-only에도 같은 평가 절차를 적용했다(Table 8).

**Table 8. 전체 Source 조합에서의 Test 성능과 적응 안정성.** 전이 피해 \\(D_G\\)와 전이 이득 \\(B_G\\)는 식 (6)의 전체 조건 평균이며, Seed SD는 구성별 세 적응 Test AP의 표준편차(분모 3)를 평균한 값이다. 큰 음수 전이는 \\(G<-0.005\\)인 실행 수, 안정적 양의 전이는 세 적응 seed 모두 \\(G>0\\)인 구성 수다. 전체 행은 네 Target에 같은 비중을 부여한다.

'''+table8(summary)+'\n\n'
    section53+=(f"**전이 피해와 이득.** 평균 Test AP는 Fine-tuning {ft['ap']:.4f}, TFRA {tf['ap']:.4f}로 "
        f"차이는 {100*(tf['ap']-ft['ap']):+.2f} percentage points였다. 평균 음수 전이 피해는 {change(tf['D_G'],ft['D_G'])}. "
        f"평균 양수 전이 이득은 {ft['B_G']:.6f}에서 {tf['B_G']:.6f}로 변했다. "
        f"큰 음수 전이는 각각 {ft['large_negative']}/372회와 {tf['large_negative']}/372회였다. "
        f"Fine-tuning의 Test 음수 전이 {p['negative']}조건 중 {p['negative_to_positive']}조건이 TFRA에서 양수 전이로 전환됐고, "
        f"Test 양수 전이 {p['positive']}조건 중 {p['positive_retained']}조건이 양수 전이를 유지했다. ")
    if p['positive']:
        section53+=(f"Fine-tuning의 Test 양수 전이 조건 집합 전체에서 평균 전이 이득은 "
            f"{p['original_positive_gain']:.6f}에서 {p['tfra_on_original_positive_gain']:.6f}로 변했다. "
            '이 조건부 집계에는 TFRA에서 음수 전이로 바뀐 조건도 포함한다. ')
    section53+='이 분류는 평가 후 전이 양상을 기술하기 위한 것이며 모델 선택에는 사용하지 않았다.\n\n'
    section53+='**기존 적응 방법과의 비교.** '
    for m in ['lp_ft','refine']:
        r=overall[m];pp=pairs[m]
        wins=sum(summary['tfra',t]['ap']>summary[m,t]['ap'] for t in TARGETS)
        section53+=(f"{LABEL[m]} 대비 TFRA의 평균 Test AP 차이는 {100*(tf['ap']-r['ap']):+.2f} percentage points였으며, "
            f"{wins}/4개 Target의 평균과 {pp['wins']}/372개 실행에서 TFRA의 AP가 높았다. "
            f"평균 음수 피해는 {change(tf['D_G'],r['D_G'])}. 양수 이득은 {r['B_G']:.6f}에서 {tf['B_G']:.6f}로 변했다. ")
    section53+='방법별 Target-only도 달라지므로, 전이 지표는 절대 Test AP와 함께 해석한다.\n\n'
    section53+=(f"**적응 seed에 걸친 결과.** Fine-tuning 대비 TFRA의 평균 Seed SD는 {change(tf['seed_sd'],ft['seed_sd'])}. "
        f"세 seed 모두 양의 전이를 보인 구성은 {ft['stable_positive']}/124개와 {tf['stable_positive']}/124개였다. "
        f"TFRA의 세 seed 평균 AP는 {p['mean_ap_improved']}/124구성에서, 관측한 세 seed 중 최저 AP는 "
        f"{p['min_ap_improved']}/124구성에서 Fine-tuning보다 높았다. 이 수치는 동일한 사전학습 seed 7에서 "
        '적응 seed 7·17·37을 평가한 범위의 기술통계다.\n\n')
    section53+='**효과의 해석.** 각 Target에서 절대 예측 성능의 변화와 Target-only 대비 기여를 구분한다. '
    for t in TARGETS:
        a,b=summary['finetune',t],summary['tfra',t]
        section53+=(f"{TARGET_LABEL[t]}의 TFRA−Fine-tuning 차이는 전이 AP {b['ap']-a['ap']:+.4f}, "
                    f"Target-only AP {b['target_only_ap']-a['target_only_ap']:+.4f}였다. ")
    section53+=(f"Source 추가 손상 \\(D_C\\)는 Fine-tuning {composition['finetune']['D_C']:.6f}, "
                f"TFRA {composition['tfra']['D_C']:.6f}였다(Appendix B). "
                'Target-only 대비 전이 효과와 Source 확장에 따른 변화는 각각의 기준으로 평가한다.\n\n')
    path=ROOT/'docs/claim.md';before_digest=digest(path);text=path.read_text()
    a=text.index('### 5.3.');b=text.index('### 5.4.',a);text=text[:a]+section53+text[b:]
    # Table 9 and all its empirical interpretations are regenerated, including dominance.
    a=text.index('**Table 9.');b=text.index('## References',a)
    assert not re.search(r'^#{2,4} ',text[a:b],re.M), 'Unexpected section after Table 9: manual review required'
    tfp=panel['tfra'];front=[LABEL[m] for m in PANEL if panel[m]['pareto']]
    body=('**Table 9. 예측층 적합과 Encoder 갱신 방식의 Test 비교.** 14개 Source 구성의 저장 모델을 '
          'Validation 선택 상태 그대로 Test에서 평가했다. Target-only와 전이 AP를 함께 제시하며, '
          '평균 전이 이득은 \\(\\bar G=B_G-D_G\\)다. 모든 값은 네 Target에 동일 비중을 부여한다.\n\n'+table9(panel)+'\n\n')
    for m in ['frozen','warmstart','joint','batch']:
        r=panel[m]
        body+=(f"{LABEL[m]} 대비 TFRA의 전이 Test AP 차이는 {tfp['ap']-r['ap']:+.4f}, "
               f"음수 피해 차이는 {tfp['D_G']-r['D_G']:+.6f}, 양수 이득 차이는 {tfp['B_G']-r['B_G']:+.6f}였다. ")
    body+='배치 단위 방식과의 비교에는 재적합 주기와 적합 표본 범위의 차이가 함께 반영된다.\n\n'
    body+='![Test 예측 성능과 전이 유용성의 파레토 비교](../plots/outputs/tfra_pareto_test.pdf)\n\n'
    body+=('**Figure 2. 진단 패널의 Test 성능과 전이 유용성.** Table 9의 Target별 동일 비중 평균을 '
           '세 목적의 쌍별 평면에 표시했다. 값은 AP percentage points 단위다. 점선은 각 평면의 '
           '비지배 점을 연결한 시각적 안내선이며, 점 사이 성능의 실현 가능성을 뜻하지 않는다. '
           '세 목적의 공동 판정은 Appendix C.3에 제시한다.\n\n')
    body+='**Test에서의 파레토 관계.** 세 목적에서 비지배인 방법은 '+', '.join(front)+'이다. '
    dominated=[LABEL[m] for m in PANEL if dominates(tfp,panel[m])]
    if dominated:body+='TFRA는 '+', '.join(dominated)+'를 지배한다. '
    if tfp['dominated_by']:body+='TFRA는 '+', '.join(LABEL[m] for m in tfp['dominated_by'])+'에 지배된다. '
    body+='이 관계는 현재 진단 패널의 Test 평균에서 계산했으며, 방법의 유일한 최적성이나 미관측 Graph에서의 지배 관계를 뜻하지 않는다.\n\n'
    text=text[:a]+body+text[b:]
    # Keep the discovery experiment on its actual split, explicitly named at first use.
    text=re.sub(r'^### 5\.2\.[^\n]*',
                '### 5.2. Transfer Damage and Adaptation Dependence (Validation)',text,count=1,flags=re.M)
    text=text.replace('먼저 전이 손상의 범위를 전체 Source 조합에서 확인한다.',
                      '개발 단계에서는 전체 Source 조합의 Validation 평가로 전이 손상의 범위를 조사했다.')
    text=re.sub(r'\*\*Table 1\.[^\n]*\*\*',
                '**Table 1. 전이 손상 진단을 위한 전체 Source 조합 Validation 실험 설정.**',text,count=1)
    text=text.replace('평가는 시간순 분할의 Validation에서 수행했으며 Test는 사용하지 않았다.',
                      'Table 1과 §5.2의 진단 평가는 시간순 분할의 Validation에서 수행했다.')
    text=text.replace('다음으로 같은 Source 구성에서 사전학습과 적응의 영향을 분리한다.',
                      '같은 개발 단계의 Validation 평가에서 사전학습과 적응의 영향을 분리한다.')
    setup='**평가 분할의 구분.** §5.2와 Table 2–7은 현상 발견과 방법 개발에 사용한 Validation 분석이다. '
    setup+=('§5.3–5.4와 Table 8–9는 기존 학습에서 Validation AP로 선택해 저장한 모델의 Test 결과다. '
            '해당 모델 목록과 파일 해시를 Test 전에 고정했으며, Test에 따른 epoch·Hyperparameter 재선택이나 '
            '예측층 재적합은 수행하지 않았다. Test는 시간순 마지막 15%이며, seed 200007의 공통 음성 후보를 사용한다. '
            'History는 각 Test 배치 시작 이전의 관측 이벤트로 제한한다. 기존 Validation 비교표는 '
            '`experiments/target_adaptation/outputs/test_v1/before_test_claim.md`에 보존한다.\n\n')
    marker='**적응 방법 비교.**'
    if '**평가 분할의 구분.**' not in text:text=text.replace(marker,setup+marker,1)
    # Introduction uses the same Test summaries, without assuming the old ordering survives.
    intro=(f"평가는 각 방법과 동일한 적응 절차의 Target-only를 기준으로 음수 전이 피해와 양수 전이 이득을 함께 측정한다. "
        f"사전학습 seed 7의 전체 Source 조합과 세 적응 seed의 372회 전이를 독립 Test에서 평가한 결과, "
        f"평균 AP는 Fine-tuning {ft['ap']:.4f}, LP-FT {overall['lp_ft']['ap']:.4f}, REFINE {overall['refine']['ap']:.4f}, "
        f"TFRA {tf['ap']:.4f}였다. Fine-tuning 대비 TFRA의 평균 음수 전이 피해는 {change(tf['D_G'],ft['D_G'])}. "
        f"TFRA는 Fine-tuning의 Test 음수 전이 {p['negative']}조건 중 {p['negative_to_positive']}조건을 양수 전이로 전환했고, "
        f"양수 전이 {p['positive']}조건 중 {p['positive_retained']}조건에서 양수 전이를 유지했다. "
        f"평균 Seed SD는 {ft['seed_sd']:.6f}에서 {tf['seed_sd']:.6f}로 변했다. "
        '14개 Source 구성의 대조실험도 동일한 Validation 선택 원칙으로 Test에서 평가하여, '
        '예측층 적합과 Encoder 갱신 방식에 따른 절대 AP·전이 피해·이득을 비교했다. '
        '이 최종 비교는 개발 단계의 Validation 관측과 평가 분할을 구분한다.')
    text,n=re.subn(r'평가는 각 방법과 동일한 적응 절차의 Target-only를 기준으로[^\n]+',lambda m:intro,text,count=1)
    assert n==1
    text=text.replace('고정된 사전학습 seed의 전체 Source 조합에서 Fine-tuning·LP-FT·REFINE과 비교하고,',
                      '고정된 사전학습 seed의 전체 Source 조합을 Test에서 Fine-tuning·LP-FT·REFINE과 비교하고,')
    # Appendix B: recompute all composition and stability claims on Test.
    a=text.index('### B.2.');b=text.index('## Appendix C.',a)
    assert len(re.findall(r'^### ',text[a:b],re.M))==2, 'Unexpected Appendix B section: manual review required'
    appendix=(f"### B.2. Test Improvements across Source Compositions\n\n"
        f"TFRA는 전체 372회 중 {p['wins']}회({100*p['wins']/372:.1f}%)에서 Fine-tuning보다 높은 Test AP를 보였다. "
        f"124개 Source 구성 중 {p['all_seed_ap_improved']}개에서는 세 적응 seed 모두 AP가 상승했고, "
        f"{p['seed_sd_reduced']}개에서는 Seed SD가 감소했다. Source 수별 Seed SD는 다음과 같다.\n\n"
        '| Source 수 | Fine-tuning | TFRA |\n|---:|---:|---:|\n')
    for k in range(1,6):appendix+=f"| {k} | {count_sd['finetune'][k]:.6f} | {count_sd['tfra'][k]:.6f} |\n"
    appendix+='\n### B.3. Test Marginal Effects of Adding a Source\n\n'
    appendix+=('비어 있지 않은 기존 Source 집합에 하나의 Source를 추가하는 방법당 900개 비교를 Test AP로 계산했다. '
        '평균 추가 기여와 평균 추가 손상은 다음과 같다.\n\n'
        '| 방법 | AP 상승 / 900 | 평균 C | 평균 추가 손상 D_C |\n|---|---:|---:|---:|\n')
    for m in MAIN:
        r=composition[m];appendix+=f"| {LABEL[m]} | {r['positive']} | {r['mean']:+.6f} | {r['D_C']:.6f} |\n"
    appendix+='\nTFRA의 Source 수별 상승 빈도는 '+', '.join(
        f"{k}→{int(k)+1}에서 {v['positive']}/{v['total']}회" for k,v in composition['tfra']['by_count'].items())+'였다. '
    appendix+=('추가 비교들은 동일한 체크포인트와 Test 표본을 공유하는 기술통계다. 총 사전학습 업데이트 60회가 '
        '고정되어 Source 수에 따라 Source당 노출량이 달라진다. Source 추가 효과와 Target-only 대비 전이 효과는 '
        '각각의 기준으로 해석한다.\n\n')
    text=text[:a]+appendix+text[b:]
    text=text.replace('각 방법의 전이 이득은 해당 방법의 Target-only를 차감하여 계산한다. Target마다 Source 구성 수가 다르므로,',
        '각 방법의 전이 이득은 해당 방법의 Test AP에서 같은 방법의 Target-only Test AP를 차감하여 계산한다. Target마다 Source 구성 수가 다르므로,')
    text=text.replace('Table 9의 절대 전이 AP, 평균 음수 전이 피해', 'Table 9의 절대 전이 Test AP, 평균 음수 전이 피해')
    a=text.index('Fine-tuning은 나머지 다섯 방법 각각에 지배되며,',text.index('### C.3.')) if 'Fine-tuning은 나머지 다섯 방법 각각에 지배되며,' in text[text.index('### C.3.'):] else -1
    if a>=0:
        end=text.find('\n\n',a);end=len(text) if end<0 else end
        last='Test 평균에서 비지배인 방법은 '+', '.join(front)+'이다. '
        for m in PANEL:
            if panel[m]['dominated_by']:
                last+=LABEL[m]+'은 '+', '.join(LABEL[v] for v in panel[m]['dominated_by'])+'에 지배된다. '
        last+=('이 분석은 14개 Source 구성과 세 적응 seed에서 관측한 Test 평균의 지배 관계를 기술한다. '
               '별도의 가중 합이나 사후 임계값을 도입하지 않았으며, Validation에서 계산한 파레토 판정과 구분한다.')
        text=text[:a]+last+text[end:]
    assert digest(path)==before_digest, 'Manuscript changed during update; rerun against the latest document'
    original=path.read_text()
    for start,end in [('## 2. Related Work','## 5. Experiments'),('## References','## Appendix B.')]:
        assert original[original.index(start):original.index(end)]==text[text.index(start):text.index(end)], \
            'Protected manuscript content changed'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'claim_test_draft.md').write_text(text)
    (out/'claim_test_draft.diff').write_text(''.join(difflib.unified_diff(
        path.read_text().splitlines(keepends=True),text.splitlines(keepends=True),
        fromfile='docs/claim.md',tofile='claim_test_draft.md')))
    atomic_json(out/'manuscript_update.json',dict(status='draft_only',claim_updated=False,
        original_sha256=before_digest,draft_sha256=digest(out/'claim_test_draft.md'),
        review_required='Inspect all changed sections and preserved sections before applying.'))
    assert digest(path)==before_digest


def report(out, update_claim=False):
    out=Path(out);freeze=read(out/'test_freeze.json');rows=records(out,'test')
    assert len(rows)==len(freeze['models'])==1752
    # Recheck every persisted prediction and equality of evaluation candidates.
    candidates={}
    expected={r['job']['id']:r for r in freeze['models']}
    assert {r['job_id'] for r in rows}==set(expected)
    for r in rows:
        saved=read(out/'test'/r['job_id']/'result.json');frozen=expected[r['job_id']]
        assert saved['job']==frozen['job'] and saved['best_sha256']==frozen['best_sha256']
        assert digest(ROOT/frozen['best'])==frozen['best_sha256']
        with np.load(out/'test'/r['job_id']/'test_predictions.npz') as a:
            assert abs(average_precision_score(a['labels'].ravel(),a['scores'].ravel())-r['ap'])<1e-12
            keys=['event_ids','src','dst','time','negatives','labels']
            if r['target'] not in candidates:candidates[r['target']]={k:a[k].copy() for k in keys}
            else:
                for k in keys:np.testing.assert_array_equal(candidates[r['target']][k],a[k])
    main=[r for r in rows if r['cohort']=='legacy_powerset'];assert len(main)==1536
    tr,ss=summarize(main);assert len(tr)==1488
    summary={(r['method'],r['target']):r for r in ss}
    plan=read(BASE/'outputs/readout_panel_v2/plan.json')
    cases={(j['target'],'+'.join(sorted(j['sources'])),j['adaptation_seed']) for j in plan['jobs']}
    panel_rows=[dict(r,cohort='legacy_panel') for r in rows if r['cohort']=='legacy_panel' or
        (r['method'] in {'finetune','tfra'} and (r['target'],r['sources'],r['adaptation_seed']) in cases)]
    assert len(panel_rows)==324
    pt,ps=summarize(panel_rows);panel={r['method']:r for r in ps if r['target']=='target_equal_mean'}
    for m,r in panel.items():
        r['dominated_by']=[n for n in PANEL if dominates(panel[n],r)];r['pareto']=not r['dominated_by']
    pairs={m:paired(tr,m) for m in ['finetune','lp_ft','refine']}
    edges,composition=additions(tr)
    count_sd={}
    for m in MAIN:
        count_sd[m]={}
        for k in range(1,6):
            groups=defaultdict(list)
            for r in tr:
                if r['method']==m and len(r['sources'].split('+'))==k:groups[r['target'],r['sources']].append(r['ap'])
            assert all(len(v)==3 for v in groups.values())
            count_sd[m][k]=float(np.mean([np.std(v) for v in groups.values()]))
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    for name,values in [('test_runs',rows),('test_transfer',tr),('test_summary',ss),('panel_transfer',pt),
                        ('panel_summary',ps),('source_additions',edges)]:csv_write(analysis/(name+'.csv'),values)
    result=dict(main=ss,panel=list(panel.values()),paired=pairs,composition=composition,count_sd=count_sd,
        split='test',model_selection='best Validation AP',verification='1752 stored predictions and shared candidates')
    atomic_json(analysis/'summary.json',result)
    lines=['# Existing-model Test evaluation','',
        'All models were selected on Validation before Test. No retraining or Test model selection.',
        '',table8(summary),'',table9(panel),'', 'All values are Test measurements.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from plots.tfra_pareto_test import render
    render(panel)
    if update_claim:manuscript(out,summary,panel,pairs,composition,count_sd)
    atomic_json(out/'verification.json',dict(passed=True,models=1752,main_transfer=1488,panel_transfer=252,
        split='test',checks=['prediction AP','paired Test candidates','complete prespecified model set',
                             'method-matched Target-only','recomputed Pareto without expected ordering']))
    return result
