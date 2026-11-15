"""Differentiable Target-fitted linear readout; model architecture is unchanged."""
import numpy as np
import torch


def hidden(model, data, ids, negatives):
    n=len(ids);enc=model.encoders[model.target]
    u=np.tile(data.src[ids],2);v=np.r_[data.dst[ids],negatives[ids]];t=np.tile(data.time[ids],2)
    a,b=enc.compute_src_dst_node_temporal_embeddings(u,v,t,num_neighbors=model.history)
    head=model.heads[model.target]
    return head[1](head[0](torch.cat((a,b),-1)))


def labels(n, device, dtype):
    return torch.cat((torch.ones(n,device=device,dtype=dtype),-torch.ones(n,device=device,dtype=dtype)))


def ridge(x,y,lam=0.001):
    # Double precision solve with differentiable casts, unpenalized intercept.
    dtype=x.dtype;x=x.double();y=y.double()
    mx=x.mean(0);my=y.mean();xc=x-mx;yc=y-my
    matrix=xc.T@xc/len(x)+lam*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
    w=torch.linalg.solve(matrix,xc.T@yc/len(x));b=my-mx@w
    return w.to(dtype),b.to(dtype)


@torch.no_grad()
def refit(model,data,negatives,cfg):
    model.bind(data);model.eval();xs=[];ys=[]
    for native in data.batches(data.indices(0),cfg['batch_size']):
        model.begin_batch(native)
        for offset in range(0,len(native),cfg['microbatch']):
            ids=native[offset:offset+cfg['microbatch']]
            x=hidden(model,data,ids,negatives)
            xs.append(x);ys.append(labels(len(ids),x.device,x.dtype))
    x=torch.cat(xs);y=torch.cat(ys);w,b=ridge(x,y)
    model.heads[model.target][2].weight.copy_(w[None]);model.heads[model.target][2].bias.copy_(b.reshape(1))
    return dict(fit_examples=len(x),ridge_lambda=0.001,fit_mse=float(((x@w+b-y)**2).mean()))
