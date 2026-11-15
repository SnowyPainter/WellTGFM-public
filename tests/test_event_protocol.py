"""Protect temporal visibility, shared query budgets, and hardware launch policy."""
import dataclasses,unittest
from unittest.mock import patch
import numpy as np
from etl.prepare_benchmark import CTDG,TKG,temporal_split,prefix
from experiments.common.data import Events,adjacency,CutoffSampler
from models.baselines.ctdg.sampling import NeighborSampler
from experiments.common.runtime import gpu_guard


def fixture():
    n=30;t=np.repeat(np.arange(10,dtype=float),3)+1
    return Events(src=np.ones(n,np.int64),dst=np.tile([2,3,4],10),relation=np.zeros(n,np.int64),
        time=t,raw_time=t,split=temporal_split(t),label=np.ones(n),edge_id=np.arange(1,n+1),
        edge_features=np.zeros((n+1,172),np.float32),metadata={'num_nodes':6,'family':'ctdg'})


class ProtocolTests(unittest.TestCase):
    def test_eight_datasets_and_no_tgb(self):
        self.assertEqual(len(CTDG),4);self.assertEqual(len(TKG),4)
        from etl.prepare_benchmark import sources
        for name in CTDG+TKG:
            self.assertTrue(all('tgb' not in url.lower() for url,_ in sources(name)))

    def test_whole_timestamp_prefix_and_split(self):
        rows=np.zeros((10,4));rows[:,3]=[1,1,2,2,3,3,4,4,5,5]
        result=prefix(rows,10);self.assertEqual(len(result),10)
        rows=np.tile(rows,(2,1));rows=rows[np.argsort(rows[:,3],kind='stable')]
        result=prefix(rows,10);self.assertEqual(len(result),8)
        data=fixture()
        for split in (0,1):self.assertLess(data.time[data.split==split].max(),data.time[data.split==split+1].min())

    def test_batches_never_split_timestamp(self):
        d=fixture();batches=list(d.batches(np.arange(30),4))
        np.testing.assert_array_equal(np.concatenate(batches),np.arange(30))
        for a,b in zip(batches,batches[1:]):self.assertLess(d.time[a[-1]],d.time[b[0]])

    def test_negatives_exclude_simultaneous_positives(self):
        d=fixture();d.dst[::3]=5
        # Add a train-only candidate that's absent at later timestamps.
        d.dst[0]=2
        a=d.negatives(7);np.testing.assert_array_equal(a,d.negatives(7))
        for i in range(30):
            true=d.dst[(d.src==d.src[i])&(d.time==d.time[i])]
            self.assertNotIn(a[i],true)
        d=fixture()
        with self.assertRaises(ValueError):d.negatives(7)

    def test_batch_cutoff_hides_current_and_future_events(self):
        d=fixture();sampler=CutoffSampler(NeighborSampler(adjacency(d),sample_neighbor_strategy='recent',seed=7))
        sampler.cutoff=4.
        _,edges,times=sampler.get_all_first_hop_neighbors(node_ids=np.array([1]),node_interact_times=np.array([9.]))
        self.assertTrue(np.all(times[0]<4.));self.assertEqual(len(edges[0]),9)

    def test_busy_gpu_rejected_before_cuda_probe(self):
        with patch('experiments.common.runtime.subprocess.check_output',return_value='100\n'), \
             patch('torch.cuda.is_available',side_effect=AssertionError('CUDA must not initialize')):
            with self.assertRaisesRegex(RuntimeError,'GPU busy'):
                with gpu_guard('cuda:0',16):pass

    def test_cpu_never_queries_gpu(self):
        with patch('experiments.common.runtime.subprocess.check_output',side_effect=AssertionError):
            with gpu_guard('cpu',16):pass

    def test_busy_compute_rejected_even_with_free_memory(self):
        with patch('experiments.common.runtime.subprocess.check_output',side_effect=['80000\n','100\n']), \
             patch('torch.cuda.is_available',side_effect=AssertionError('CUDA must not initialize')):
            with self.assertRaisesRegex(RuntimeError,'utilization'):
                with gpu_guard('cuda:0',16):pass


if __name__=='__main__':unittest.main()
