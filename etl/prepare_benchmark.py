"""Download JODIE/TKG sources and build immutable, shared 32,768-event benchmarks."""
from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
CTDG = ('wikipedia', 'reddit', 'mooc', 'lastfm')
TKG = ('ICEWS14', 'ICEWS05-15', 'GDELT', 'ICEWS18')
TLOGIC = '9f6ffa3cc5c112226dc2537983b6bb33d4ef295e'
CYGNET = '1295cb32cb55a04c728735994f8226a11fa90056'
VERSION = 'chronological_32768_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            print('download', url, flush=True)
            with urlopen(url, timeout=45) as response, temp.open('wb') as out:
                shutil.copyfileobj(response, out)
            if temp.stat().st_size == 0: raise ValueError('Empty download')
            temp.replace(path)
            return
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2: raise
            time.sleep(1)


def sources(name):
    if name in CTDG:
        return [(f'https://snap.stanford.edu/jodie/{name}.csv', RAW/'ctdg'/f'{name}.csv')]
    if name == 'GDELT':
        base = f'https://raw.githubusercontent.com/CunchaoZ/CyGNet/{CYGNET}/data/GDELT'
        files = ['train.txt', 'valid.txt', 'test.txt', 'stat.txt']
    else:
        folder = {'ICEWS14':'icews14', 'ICEWS05-15':'icews0515', 'ICEWS18':'icews18'}[name]
        base = f'https://raw.githubusercontent.com/liu-yushan/TLogic/{TLOGIC}/data/{folder}'
        files = ['train.txt', 'valid.txt', 'test.txt', 'entity2id.json', 'relation2id.json', 'ts2id.json']
    return [(base+'/'+f, RAW/'tkg'/name/f) for f in files]


def temporal_split(times):
    cuts=[]
    for fraction in (.70,.85):
        at=int(len(times)*fraction)
        cuts.append(int(np.searchsorted(times,times[at],side='left')))
    if not 0 < cuts[0] < cuts[1] < len(times):
        raise ValueError('Need at least three sufficiently separated timestamp groups')
    split=np.full(len(times),2,dtype=np.uint8)
    split[:cuts[1]]=1;split[:cuts[0]]=0
    return split


def prefix(values, limit):
    if limit < 10: raise ValueError('max-events must be >= 10')
    if np.any(np.diff(values[:,3]) < 0):
        values=values[np.argsort(values[:,3],kind='stable')]
    n=min(limit,len(values))
    if n < len(values) and values[n-1,3] == values[n,3]:
        n=int(np.searchsorted(values[:,3],values[n,3],side='left'))
    if n == 0: raise ValueError('First timestamp exceeds event budget')
    return values[:n]


def prepare(name, max_events=32768, output=None):
    output=Path(output or ROOT/'data/benchmarks'/f'events{max_events}')
    family='ctdg' if name in CTDG else 'tkg'
    folder=output/family/name
    for url,path in sources(name):download(url,path)
    provenance=[dict(url=u,path=str(p.relative_to(ROOT)),sha256=sha(p)) for u,p in sources(name)]
    meta_path=folder/'metadata.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old['version']!=VERSION or old['max_events']!=max_events or old['sources']!=provenance:
            raise ValueError(f'Existing dataset differs: {folder}; choose a new output root')
        for filename,digest in old['artifacts'].items():
            if sha(folder/filename)!=digest: raise ValueError(f'Corrupt artifact: {folder/filename}')
        print('verified existing',name,flush=True)
        return old
    if family=='ctdg':
        # One extra row identifies a timestamp truncated by the cap.
        frame=pd.read_csv(RAW/'ctdg'/f'{name}.csv',skiprows=1,header=None,nrows=max_events+1)
        v=frame.to_numpy(dtype=np.float64)
        # Keep edge features aligned when sorting a source stream.
        v=v[np.argsort(v[:,2],kind='stable')]
        rows=np.column_stack((v[:,0],np.zeros(len(v)),v[:,1],v[:,2]))
        rows=prefix(rows,max_events);v=v[:len(rows)]
        _,s=np.unique(rows[:,0],return_inverse=True)
        _,d=np.unique(rows[:,2],return_inverse=True)
        src=s+1;dst=d+s.max()+2;rel=np.zeros(len(rows),dtype=np.int64)
        raw_features=v[:,4:].astype(np.float32)
        if raw_features.shape[1]>172: raise ValueError('Unexpected JODIE feature width')
        features=np.zeros((len(rows)+1,172),np.float32)
        features[1:,:raw_features.shape[1]]=raw_features
        labels=v[:,3].astype(np.float32)
    else:
        source=RAW/'tkg'/name
        if name=='GDELT':
            arrays=[np.loadtxt(source/f'{sp}.txt',usecols=(0,1,2,3),dtype=np.int64,ndmin=2) for sp in ['train','valid','test']]
        else:
            ent=json.loads((source/'entity2id.json').read_text())
            relations=json.loads((source/'relation2id.json').read_text())
            stamps=json.loads((source/'ts2id.json').read_text())
            arrays=[]
            for sp in ['train','valid','test']:
                records=[]
                with (source/f'{sp}.txt').open() as f:
                    for line in f:
                        a,r,b,t=line.strip().split('\t')
                        records.append((ent[a],relations[r],ent[b],stamps[t]))
                arrays.append(np.asarray(records,np.int64))
        rows=prefix(np.concatenate(arrays),max_events)
        entities=np.unique(rows[:,[0,2]])
        src=np.searchsorted(entities,rows[:,0])+1;dst=np.searchsorted(entities,rows[:,2])+1
        relations,rel=np.unique(rows[:,1],return_inverse=True)
        rel=rel+1  # 0 is padding/CTDG's untyped relation.
        features=np.zeros((len(rows)+1,172),np.float32);labels=np.ones(len(rows),np.float32)
    raw_time=rows[:,3].astype(np.float64)
    split=temporal_split(raw_time)
    train_times=raw_time[split==0];gaps=np.diff(np.unique(train_times))
    scale=float(np.median(gaps)) if len(gaps) else 1.
    times=(raw_time-raw_time[0])/max(scale,1e-12)+1.
    folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'events.npz',src=src.astype(np.int64),dst=dst.astype(np.int64),relation=rel,
             time=times,raw_time=raw_time,split=split,label=labels,edge_id=np.arange(1,len(rows)+1))
    np.save(folder/'edge_features.npy',features)
    metadata=dict(version=VERSION,dataset=name,family=family,max_events=max_events,num_events=len(rows),
        num_nodes=int(max(src.max(),dst.max()))+1,num_relations=int(rel.max())+1,
        split_counts=np.bincount(split,minlength=3).tolist(),time_origin=float(raw_time[0]),time_scale=scale,
        split_policy='chronological prefix, whole timestamps, 70/15/15; custom small-data benchmark',
        sources=provenance,artifacts={f:sha(folder/f) for f in ['events.npz','edge_features.npy']})
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    print(name,metadata['num_events'],metadata['split_counts'],flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=CTDG+TKG,default=list(CTDG+TKG))
    p.add_argument('--max-events',type=int,default=32768)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    for name in args.datasets: prepare(name,args.max_events,args.output)

if __name__=='__main__':main()
