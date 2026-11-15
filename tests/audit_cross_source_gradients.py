"""CPU audit on real source batches; isolated models, read-only trained checkpoint."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from experiments.common import protocol
from experiments.common.data import Events
from experiments.common.runtime import atomic_json,seed_all
from experiments.transfer.cross_representation import build,typed_negatives,optimizer
from models.ours.joint import shared_key


def group(k):
    return '.'.join(k.split('.')[:2]) if k.startswith(('temporal_conv_layers.','merge_layers.')) else k.split('.')[0]


def audit():
    cfg=yaml.safe_load((protocol.ROOT/'experiments/transfer/cross_representation.yaml').read_text());cfg['device']='cpu'
    torch.set_num_threads(2)
    data={n:Events.load(protocol.ROOT/cfg['data_root'],cfg['families'][n],n) for n in cfg['datasets']}
    out=protocol.ROOT/'experiments/transfer/outputs/gradient_audit_v1';out.mkdir(parents=True,exist_ok=True)
    path=protocol.ROOT/'experiments/transfer/outputs/cross_representation_v1/mooc+ICEWS14/seed7/pretrain/resume.pt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    trained=torch.load(path,map_location='cpu',weights_only=False)
    assert trained['step']==60
    records=[]
    for state in ('initial','joint_step60'):
        m=build(cfg,data,7)
        if state=='joint_step60':m.load_state_dict(trained['model'])
        original={k:v.detach().clone() for k,v in m.state_dict().items()}
        params={n:dict(e.named_parameters()) for n,e in m.encoders.items()}
        shared={k:v for k,v in params['lastfm'].items() if shared_key(k)}
        assert all(params[n][k] is p and params[n][k].data_ptr()==p.data_ptr() for n in ('mooc','ICEWS14') for k,p in shared.items())
        for name in ('mooc','ICEWS14'):
            m.load_state_dict(original);m.zero_grad(set_to_none=True);seed_all(7);m.train();m.bind(data[name])
            d=data[name];batches=list(d.batches(d.indices(0),cfg['tkg_batch_size'] if name=='ICEWS14' else cfg['batch_size']))
            batch=batches[len(batches)//2];ids=batch[:32];m.begin_batch(batch)
            negatives=typed_negatives(d,7) if name=='ICEWS14' else d.negatives(7)
            scores=m.score(ids,negatives[ids]);truth=torch.zeros_like(scores);truth[:,0]=1
            loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,truth)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            grads={k:float(p.grad.norm()) if p.grad is not None else None for k,p in shared.items()}
            private={}
            for n,enc in m.encoders.items():
                private[n]=dict(encoder=float(np.sqrt(sum(float(p.grad.square().sum()) for k,p in enc.named_parameters() if not shared_key(k) and p.grad is not None))),head=float(np.sqrt(sum(float(p.grad.square().sum()) for p in m.heads[n].parameters() if p.grad is not None))))
            assert private[name]['head']>0
            for n in data:
                if n!=name:assert private[n]['head']==0 and private[n]['encoder']==0
            if name=='ICEWS14':assert private[name]['encoder']>0
            opt=optimizer(m,cfg);optids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(optids)==len(set(optids))
            assert all(optids.count(id(p))==1 for p in shared.values())
            before={k:p.detach().clone() for k,p in shared.items()}
            torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['grad_clip']);opt.step()
            deltas={k:float((p.detach()-before[k]).norm()) for k,p in shared.items()}
            groups={g:dict(gradient_norm=float(np.sqrt(sum(v*v for k,v in grads.items() if group(k)==g and v is not None))),update_norm=float(np.sqrt(sum(v*v for k,v in deltas.items() if group(k)==g)))) for g in sorted({group(k) for k in shared})}
            assert all(v['gradient_norm']>0 and v['update_norm']>0 for v in groups.values())
            records.append(dict(state=state,source=name,loss=float(loss.detach()),events=len(ids),prediction_pairs=len(scores),shared_parameter_tensors=len(shared),same_parameter_objects=True,optimizer_duplicates=0,gradient_norms=grads,update_norms=deltas,groups=groups,private_norms=private,zero_gradient_tensors=[k for k,v in grads.items() if not v],missing_gradient_tensors=[k for k,v in grads.items() if v is None]))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    atomic_json(out/'audit.json',dict(status='passed',device='cpu',checkpoint=str(path),checkpoint_sha256=digest,checkpoint_unchanged=True,records=records))
    lines=['# 두 Source의 공유 TGAT Gradient 검사','',
           '실제 MOOC·ICEWS14 Train 중간 배치의 32개 이벤트를 각각 사용했다. 초기 모델과 공동 사전학습 60회 모델을 복제해 CPU에서 검사했으며, 학습 중인 모델·Optimizer·원본 Checkpoint는 변경하지 않았다.','',
           '| 모델 상태 | Source | 공유 계층 | Gradient L2 norm | Adam 업데이트 L2 norm |','|---|---|---|---:|---:|']
    for r in records:
        for g,v in r['groups'].items():lines.append(f"| {r['state']} | {r['source']} | {g} | {v['gradient_norm']:.6g} | {v['update_norm']:.6g} |")
    lines+=['','공유 계층의 Parameter 객체 및 storage 주소가 두 Source에서 동일함을 확인했다. 각 Source의 단독 역전파가 모든 공유 계층 그룹에 유한한 비영 Gradient를 전달했고, Adam step이 해당 계층을 실제 변경했다. Optimizer에 공유 파라미터가 중복 등록되지 않았다.',
            'Domain별 Head와 TKG 입력부는 해당 Source에서만 Gradient를 받았다. 일부 개별 Tensor의 Gradient가 0일 수 있으므로 Tensor별 원자료도 audit.json에 보존했다. 이 검사는 학습 연결의 정확성을 검증하며, 지식 축적이나 전이 이득 자체를 보장하지 않는다.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',rows=[dict(state=r['state'],source=r['source'],groups=r['groups'],zero_tensors=len(r['zero_gradient_tensors']),missing_tensors=len(r['missing_gradient_tensors']),shared_tensors=r['shared_parameter_tensors']) for r in records]),indent=2))

if __name__=='__main__':audit()
