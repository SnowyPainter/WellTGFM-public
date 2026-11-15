import unittest
import torch
from experiments.target_adaptation.readout import ridge

class RidgeTest(unittest.TestCase):
    def test_stationary_solution(self):
        torch.manual_seed(4);x=torch.randn(20,5,dtype=torch.double);y=torch.randn(20,dtype=torch.double)
        w,b=ridge(x,y);res=x@w+b-y
        torch.testing.assert_close(x.T@res/len(x)+.001*w,torch.zeros_like(w),atol=1e-10,rtol=0)
        self.assertLess(abs(float(res.mean())),1e-10)
    def test_support_gradient(self):
        torch.manual_seed(7);x=torch.randn(8,3,dtype=torch.double,requires_grad=True);y=torch.randn(8,dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda z: ridge(z,y), (x,),eps=1e-6,atol=1e-4))

class EnvelopeTest(unittest.TestCase):
    def test_profiled_risk_gradient(self):
        torch.manual_seed(9);x=torch.randn(12,4,dtype=torch.double,requires_grad=True);y=torch.randn(12,dtype=torch.double)
        w,b=ridge(x,y)
        full=((x@w+b-y)**2).mean()+.001*w.square().sum()
        grad_full=torch.autograd.grad(full,x,retain_graph=True)[0]
        partial=((x@w.detach()+b.detach()-y)**2).mean()
        grad_partial=torch.autograd.grad(partial,x)[0]
        torch.testing.assert_close(grad_full,grad_partial,atol=1e-10,rtol=1e-8)
