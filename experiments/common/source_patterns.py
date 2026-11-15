"""Source diagnostic helpers preserved from source_role_conflict/run.py."""
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.common.data import Events
from experiments.common.runtime import atomic_json

ROOT=Path(__file__).resolve().parents[2]
DATASETS=('wikipedia','reddit','mooc','lastfm')
SEEDS=(7,17,37)
VIEWS=('recurrence','recency','transition','recurrence_recency','recurrence_transition','all')


def count_bin(n):
    return '0' if n==0 else '1' if n==1 else '2-3' if n<4 else '4-7' if n<8 else '8+'


def lag_bin(n):
    return 'never' if n is None else '1' if n<=1 else '2-3' if n<4 else '4-15' if n<16 else '16+'


def features(data,negatives,batch_size=2048):
    counts=Counter();activity=Counter();last={};previous={};transitions=Counter();cursor=0
    records=[]
    def advance(cutoff):
        nonlocal cursor
        while cursor<len(data.src) and data.raw_time[cursor]<cutoff:
            end=cursor+1;t=data.raw_time[cursor]
            while end<len(data.src) and data.raw_time[end]==t:end+=1
            current=defaultdict(set)
            for i in range(cursor,end):
                u,v=int(data.src[i]),int(data.dst[i]);activity[u]+=1;counts[u,v]+=1;last[u,v]=activity[u]
                for old in previous.get(u,()):transitions[old,v]+=1
                current[u].add(v)
            previous.update(current);cursor=end
    for split in (0,1):
        for ids in data.batches(data.indices(split),batch_size):
            advance(data.raw_time[ids[0]])
            for i in ids:
                u=int(data.src[i])
                for label,v in ((1,int(data.dst[i])),(0,int(negatives[i]))):
                    repeat=counts[u,v];lag=None if (u,v) not in last else activity[u]-last[u,v]+1
                    trans=sum(transitions[old,v] for old in previous.get(u,()))
                    r,c,t=count_bin(repeat),lag_bin(lag),count_bin(trans)
                    cells=(r,c,t,r+'|'+c,r+'|'+t,r+'|'+c+'|'+t)
                    records.append((split,label,cells))
    return records


def aggregate(records):
    tables={split:{view:{} for view in VIEWS} for split in (0,1)}
    for split,label,cells in records:
        for view,cell in zip(VIEWS,cells):
            value=tables[split][view].setdefault(cell,[0,0]);value[0]+=1;value[1]+=label
    return tables


def nll(p,positive,total):
    p=float(np.clip(p,1e-9,1-1e-9));return -(positive*np.log(p)+(total-positive)*np.log1p(-p))/total


def wilson(positive,total):
    p=positive/total;z=1.96;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    width=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(center-width),float(center+width)]


def compare(tables,seed,min_count=100):
    rows=[]
    for a,b in itertools.combinations(DATASETS,2):
        for view in VIEWS:
            ta,tb=tables[a][0][view],tables[b][0][view]
            na=sum(x[0] for x in ta.values());nb=sum(x[0] for x in tb.values())
            for cell in sorted(set(ta)&set(tb)):
                an,ay=ta[cell];bn,by=tb[cell]
                if min(an,bn)<min_count:continue
                pa,pb=ay/an,by/bn
                # Discovery is strictly train-only. Same cell, strong opposite sampled-label roles.
                discovered=(pa>=.6 and pb<=.4) or (pb>=.6 and pa<=.4)
                av=tables[a][1][view].get(cell,[0,0]);bv=tables[b][1][view].get(cell,[0,0])
                # Equal SOURCE weights in the original risks, not equal cell weights.
                pooled=(ay/na+by/nb)/(an/na+bn/nb)
                row=dict(seed=seed,source_a=a,source_b=b,view=view,cell=cell,
                    a_train_n=an,b_train_n=bn,a_train_p=pa,b_train_p=pb,pooled_train_p=pooled,
                    train_discovered=discovered,a_val_n=av[0],b_val_n=bv[0],val_supported=min(av[0],bv[0])>=min_count)
                if av[0] and bv[0]:
                    va,vb=av[1]/av[0],bv[1]/bv[0]
                    ci_a,ci_b=wilson(av[1],av[0]),wilson(bv[1],bv[0])
                    direction=1 if pa>pb else -1
                    confirmed=row['val_supported'] and ((direction==1 and va>=.6 and vb<=.4) or (direction==-1 and vb>=.6 and va<=.4))
                    row.update(a_val_p=va,b_val_p=vb,a_val_wilson=ci_a,b_val_wilson=ci_b,
                        val_direction_matches=(va-vb)*direction>0,
                        val_strong_opposition=confirmed,
                        pooled_val_ce=(nll(pooled,av[1],av[0])+nll(pooled,bv[1],bv[0]))/2,
                        source_aware_val_ce=(nll(pa,av[1],av[0])+nll(pb,bv[1],bv[0]))/2)
                    row['ce_excess']=row['pooled_val_ce']-row['source_aware_val_ce']
                rows.append(row)
    return rows

