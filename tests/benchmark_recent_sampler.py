import json,time,types
from pathlib import Path
import numpy as np
import torch
from experiments.common.data import Events,adjacency
from models.baselines.ctdg.sampling import NeighborSampler
from experiments.transfer.cross_representation import build,typed_negatives
from experiments.common.runtime import seed_all
root=Path(__file__).resolve().parents[1]
data={n:Events.load(root/'data/benchmarks/events32768',f,n) for n,f in [('wikipedia','ctdg'),('ICEWS14','tkg'),('GDELT','tkg')]}
rows=[]
for name,d in data.items():
 s=NeighborSampler(adjacency(d),sample_neighbor_strategy='recent',seed=7)
 rng=np.random.default_rng(7);ids=rng.integers(0,len(d.src),32768);nodes=d.src[ids];times=d.time[ids]
 results=[]
 for method in [s._historical_neighbors_reference,s.get_historical_neighbors]:
  start=time.perf_counter();v=method(nodes,times,32);elapsed=time.perf_counter()-start
  results.append((elapsed,v))
 for a,b in zip(results[0][1],results[1][1]): np.testing.assert_array_equal(a,b)
 rows.append(dict(dataset=name,queries=len(ids),reference_seconds=results[0][0],vectorized_seconds=results[1][0],speedup=results[0][0]/results[1][0],exact=True))
torch.set_num_threads(2)
cfg=dict(target='wikipedia',history_length=32,device='cpu')
m=build(cfg,data,7)
checks=[]
for name,d in data.items():
 ids=d.indices(0)[10000:10008]
 neg=typed_negatives(d,7)[ids] if d.metadata['family']=='tkg' else d.negatives(7)[ids]
 m.bind(d);m.begin_batch(ids);m.train()
 vals=[]
 for reference in [True,False]:
  for sampler in m.samplers.values():
   sampler.base.get_historical_neighbors=types.MethodType(NeighborSampler._historical_neighbors_reference if reference else NeighborSampler.get_historical_neighbors,sampler.base)
  seed_all(777);m.zero_grad(set_to_none=True);z=m.score(ids,neg)
  y=torch.zeros_like(z);y[:,0]=1
  loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y);loss.backward()
  vals.append((z.detach().clone(),{k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}))
 torch.testing.assert_close(vals[0][0],vals[1][0],rtol=0,atol=0)
 for k in vals[0][1]:torch.testing.assert_close(vals[0][1][k],vals[1][1][k],rtol=0,atol=0)
 checks.append(dict(dataset=name,logits_and_gradients_bitwise_equal=True))
output=root/'experiments/transfer/outputs/sampler_optimization_v1';output.mkdir(exist_ok=True)
result=dict(sampler_benchmark=rows,model_checks=checks)
(output/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
