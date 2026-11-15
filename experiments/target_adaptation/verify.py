"""Verify retained results, inputs, and optionally saved-model inference."""
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from experiments.target_adaptation.runtime import ROOT,HERE,load_inputs,digest,array_hash,build
from experiments.common.runtime import evaluate,gpu_guard,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',action='store_true')
    args=parser.parse_args()
    jobs,cfgs,data=load_inputs()
    torch.set_num_threads(2)
    archive=ROOT/json.loads((HERE/'inputs/provenance.json').read_text())['archive']
    protocols=json.loads((HERE/'outputs/original_protocol.json').read_text())
    assert cfgs==protocols['configs'] and protocols['test_used'] is False
    hashes={};records=[]
    with gpu_guard('cuda:0',0) if args.inference else nullcontext():
        for method in ['finetune','ridge_epoch']:
            for job in jobs:
                folder=HERE/'outputs'/method/job['job_id']
                result=json.loads((folder/'result.json').read_text())
                manifest=json.loads((folder/'manifest.json').read_text())
                assert result['status']=='complete' and result['method']==method
                assert result['job']==manifest['job']
                assert all(result['job'][k]==v for k,v in job.items() if k!='checkpoint')
                assert manifest['config']==cfgs[job['target']]
                if job['sources']:
                    assert digest(ROOT/job['checkpoint'])==job['checkpoint_sha256']==result['checkpoint_sha256']
                d=data[job['target']];ids=d.indices(1);neg=d.negatives(100007)
                with np.load(folder/'validation_predictions.npz') as a:
                    for key,value in dict(event_ids=d.edge_id[ids],src=d.src[ids],dst=d.dst[ids],time=d.raw_time[ids],negatives=neg[ids],labels=np.tile([1,0],(len(ids),1))).items():
                        assert np.array_equal(a[key],value),(method,job['job_id'],key)
                    assert array_hash(a)==result['evaluation_sha256']
                    assert hashes.setdefault(job['target'],array_hash(a))==array_hash(a)
                    expected=a['scores'].ravel().copy()
                    assert np.isfinite(expected).all()
                    assert abs(average_precision_score(a['labels'].ravel(),expected)-result['ap'])<1e-10
                aps=[h['validation']['ap'] for h in result['history']]
                assert len(aps)==5 and result['best_epoch']==1+int(np.argmax(aps))
                assert result['ap']==max(aps)
                # The archived directory contains the exact source records, including symlinked baselines.
                original=archive/'experiments/adaptation_screening/outputs/profiled_ablation_v1'/method/job['job_id']
                for name in ['result.json','manifest.json','best.pt','validation_predictions.npz']:
                    assert digest(folder/name)==digest(original/name),(method,job['job_id'],name)
                record=dict(method=method,job_id=job['job_id'],ap=result['ap'],original_files_unchanged=True)
                if args.inference and method=='ridge_epoch':
                    state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                    assert state['epoch']==result['best_epoch']
                    model=build(cfgs[job['target']],data,job['adaptation_seed'])
                    model.load_state_dict(state['model'],strict=True);model.bind(d)
                    metrics,scores=evaluate(model,d,1,neg,cfgs[job['target']])
                    np.testing.assert_allclose(scores,expected,rtol=1e-5,atol=1e-6)
                    assert abs(metrics['ap']-result['ap'])<1e-6
                    record['max_score_difference']=float(np.max(np.abs(scores-expected)))
                    del model,state;torch.cuda.empty_cache()
                records.append(record)
                print(json.dumps(record),flush=True)
    atomic_json(HERE/'outputs'/('inference_verification.json' if args.inference else 'verification.json'),dict(status='passed',checked_runs=len(records),inference_runs=sum('max_score_difference' in r for r in records),test_used=False,records=records))

if __name__=='__main__':main()
