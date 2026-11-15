"""Select observed failures from saved baselines; no training or checkpoint writes."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/transfer/outputs/cross_representation_two_tkg_v1'
OUT = BASE / 'analysis/mean_failure_analysis'
SEEDS = (7, 17, 37)


def main():
    rows = []
    for target in ('wikipedia', 'reddit', 'mooc', 'lastfm'):
        summary = json.loads((BASE / target / 'summary.json').read_text())
        for entry in summary['rows']:
            for condition in summary['conditions']:
                rows.append(dict(target=target, seed=entry['seed'],
                    sources=[] if condition == 'target_only' else condition.split('+'),
                    baseline_bce=entry[condition], target_only=entry['target_only']))
    lookup = {(r['target'], frozenset(r['sources']), r['seed']): r for r in rows}
    assert len(lookup) == 384
    # Cross-check the original result files, not the arithmetic candidate columns.
    for r in rows:
        directory = '+'.join(r['sources']) if r['sources'] else 'target_only'
        result = BASE / r['target'] / directory / f"seed{r['seed']}" / 'target/result.json'
        if not result.exists() and not r['sources']:
            result = BASE / r['target'] / directory / f"seed{r['seed']}" / 'result.json'
        original = json.loads(result.read_text())
        assert abs(original['metrics']['ap'] - r['baseline_bce']) < 1e-12, result
    source_edges = json.loads((BASE / 'analysis/source_interactions/edges.json').read_text())
    source_lookup = {(e['target'], frozenset(e['sources'].split('+')), e['added_source'], e['seed']): e for e in source_edges}
    grouped = defaultdict(list)
    all_edges = []
    for e in source_edges:
        t, s, added, seed = e['target'], frozenset(e['sources'].split('+')), e['added_source'], e['seed']
        before, after = lookup[t, s, seed], lookup[t, s | {added}, seed]
        single = lookup[t, frozenset([added]), seed]
        old = source_lookup[t, s, added, seed]
        delta = after['baseline_bce'] - before['baseline_bce']
        assert abs(delta - old['target_delta']) < 1e-12
        item = dict(target=t, sources='+'.join(before['sources']), added=added, seed=seed,
                    before=before['baseline_bce'], after=after['baseline_bce'], delta=delta,
                    target_only=before['target_only'], added_alone=single['baseline_bce'],
                    before_gain=before['baseline_bce']-before['target_only'],
                    added_alone_gain=single['baseline_bce']-before['target_only'],
                    after_gain=after['baseline_bce']-after['target_only'],
                    all_existing_source_ap_up=all(v > 0 for v in old['source_deltas'].values()))
        all_edges.append(item)
        grouped[t, item['sources'], added].append(item)
    summaries = []
    for (t, s, a), es in grouped.items():
        es.sort(key=lambda e: e['seed'])
        assert tuple(e['seed'] for e in es) == SEEDS
        summaries.append(dict(target=t, sources=s, added=a, before=mean(e['before'] for e in es),
                              after=mean(e['after'] for e in es), delta=mean(e['delta'] for e in es),
                              delta_std=stdev(e['delta'] for e in es),
                              negative_seeds=sum(e['delta'] < 0 for e in es),
                              before_positive_seeds=sum(e['before_gain'] > 0 for e in es),
                              added_useful_but_harmful_seeds=sum(e['added_alone_gain'] > 0 and e['delta'] < 0 for e in es),
                              both_useful_but_harmful_seeds=sum(e['added_alone_gain'] > 0 and e['before_gain'] > 0 and e['delta'] < 0 for e in es),
                              source_up_target_down_seeds=sum(e['all_existing_source_ap_up'] and e['delta'] < 0 for e in es),
                              deltas=[e['delta'] for e in es]))
    summaries.sort(key=lambda e: (-e['negative_seeds'], e['delta']))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, records in [('edges', all_edges), ('groups', summaries)]:
        (OUT / f'{name}.json').write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
        with (OUT / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    nonempty = [r for r in rows if r['sources']]
    counts = dict(conditions=len(rows), nonempty_conditions=len(nonempty),
                  target_only_declines=sum(r['baseline_bce'] < r['target_only'] for r in nonempty),
                  addition_comparisons=len(all_edges), addition_declines=sum(e['delta'] < 0 for e in all_edges),
                  groups=len(summaries), all_seed_declines=sum(e['negative_seeds'] == 3 for e in summaries),
                  independently_useful_addition_harm=sum(e['added_alone_gain'] > 0 and e['delta'] < 0 for e in all_edges),
                  both_useful_addition_harm=sum(e['added_alone_gain'] > 0 and e['before_gain'] > 0 and e['delta'] < 0 for e in all_edges),
                  all_source_up_target_down=sum(e['all_existing_source_ap_up'] and e['delta'] < 0 for e in all_edges))
    (OUT / 'summary.json').write_text(json.dumps(counts, indent=2)+'\n')
    lines = ['# 기존 평균 공동학습의 실패 조건', '',
             '저장된 Validation AP만 분석했다. 새 학습·평가·Test 사용은 없다. 원본 384개 결과와 AP를 대조했다.', '',
             '전체 Power Set은 **Source BCE + 균등 순차 공동학습**이다. 평균 Source 위험을 겨냥한 순차 Adam 실행이며, 매 step 동시 평균 Gradient를 계산한 실험은 아니다.', '',
             '총 60회 사전학습 업데이트, Target BCE 적응 5 epoch, 시드 7·17·37. Source 수가 늘면 Source당 업데이트는 줄어든다. 여기서 실패는 관측된 전이 성능 하락을 뜻하며 손실 결합식만의 인과 효과로 해석하지 않는다.', '',
             '## 전체 결과', '']
    labels = ['전체 학습 조건', '사전학습 조건', 'Target-only 대비 하락', 'Source 추가 비교',
              'Source 추가 후 하락', '추가 구성쌍', '세 시드 모두 하락한 구성쌍',
              '추가 Source가 단독으로 유용한데 추가 후 하락',
              '기존 집합과 추가 Source가 각각 유용한데 추가 후 하락',
              '기존 Source AP가 모두 상승했는데 Target AP 하락']
    lines += [f'- {label}: {v}' for label, v in zip(labels, counts.values())]
    lines += ['', '900건은 300개 추가 구성쌍 × 3시드이며 체크포인트를 공유하는 대응 비교다. 독립 실험 900회가 아니다.', '']

    def table(title, selected):
        lines.extend([f'## {title}', '', '| Target | 기존 Source | 추가 Source | 기존 AP | 추가 후 AP | 추가 효과 평균 ± 표준편차 | 시드별 추가 효과 (7 / 17 / 37) |', '|---|---|---|---:|---:|---:|---|'])
        for e in selected:
            ds = ' / '.join(f'{d:+.6f}' for d in e['deltas'])
            lines.append(f"| {e['target']} | {e['sources']} | {e['added']} | {e['before']:.6f} | {e['after']:.6f} | {e['delta']:+.6f} ± {e['delta_std']:.6f} | {ds} |")
        if not selected:
            lines.append('해당 조건 없음.')
        lines.append('')

    # One strongest all-seed decline per Target, rather than only one target's extremes.
    selected = []
    for t in ('wikipedia', 'reddit', 'mooc', 'lastfm'):
        candidates = [e for e in summaries if e['target'] == t and e['negative_seeds'] == 3]
        if candidates:
            selected.append(min(candidates, key=lambda e: e['delta']))
    table('세 시드 모두 하락한 사례: Target별 평균 하락이 가장 큰 구성', selected)
    useful = sorted(summaries, key=lambda e: (-e['both_useful_but_harmful_seeds'], -e['added_useful_but_harmful_seeds'], e['delta']))
    table('기존 집합과 추가 Source가 각각 유용한데 결합하면 하락한 사례', [e for e in useful if e['both_useful_but_harmful_seeds'] > 0][:6])
    lines += ['위 사례의 조건 충족 반복 수:', '']
    for e in [e for e in useful if e['both_useful_but_harmful_seeds'] > 0][:6]:
        lines.append(f"- {e['target']}: {e['sources']} + {e['added']} — {e['both_useful_but_harmful_seeds']}/3. 전체 평균과 개별 반복의 조건 충족은 구분한다.")
    lines += ['', '대표 구성의 단독 유용성 확인:', '',
              '| Target | 기존 Source | 추가 Source | 시드 | Target-only | 기존 집합 AP | 추가 Source 단독 AP | 결합 AP |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    representative_keys = {(e['target'], e['sources'], e['added']) for e in useful[:2]}
    for e in all_edges:
        if (e['target'], e['sources'], e['added']) in representative_keys:
            lines.append(f"| {e['target']} | {e['sources']} | {e['added']} | {e['seed']} | {e['target_only']:.6f} | {e['before']:.6f} | {e['added_alone']:.6f} | {e['after']:.6f} |")
    lines += ['', '## 해석', '',
              '기존 균등 공동학습에서 Source 추가에 따른 전이 성능 하락은 이미 관측되며, 세 시드 모두 반복되는 구성도 있다. 단독으로 유용한 Source를 추가해도 하락하는 경우를 별도로 식별했다.', '',
              '기존 Source AP 개선과 Target AP 하락이 동반되는 사례는 Source 예측 성능을 전이 유용성의 대리 지표로 사용할 때의 한계를 보여준다. AP 개선을 Source Loss 감소로 치환하지 않는다. Gradient 충돌이나 특정 원인을 이 분석으로 확정하지 않는다.', '',
              '실패 사례는 기존 Validation 결과를 보고 선택한 탐색용 구성이다. 새 방법의 전체 효과를 평가하는 표본으로 일반화하지 않는다. 모든 구성과 시드별 값은 groups.csv와 edges.csv에 남겼다.', '',
              '재현: `/opt/conda/bin/python -m experiments.transfer.analyze_mean_failures`', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(counts, indent=2))
    print(OUT / 'README.md')


if __name__ == '__main__':
    main()
