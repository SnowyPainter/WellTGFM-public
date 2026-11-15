import numpy as np
from models.baselines.ctdg.sampling import NeighborSampler
from experiments.common.data import CutoffSampler


def test_recent_exact_boundaries_and_padding():
    rng = np.random.default_rng(19)
    adj = [[], []] + [[(int(rng.integers(1, 40)), j+1, float(rng.integers(0, 80)))
                       for j in range(int(rng.integers(0, 500)))] for _ in range(40)]
    sampler = NeighborSampler(adj, sample_neighbor_strategy='recent', seed=7)
    nodes = rng.integers(0, len(adj), 5000)
    times = rng.choice(np.r_[np.arange(-1, 82), -np.inf, np.inf, np.nan], len(nodes))
    for width in (1, 32, 600):
        expected = sampler._historical_neighbors_reference(nodes, times, width)
        actual = sampler.get_historical_neighbors(nodes, times, width)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype
    wrapped = CutoffSampler(sampler); wrapped.cutoff = 24.
    for a, b in zip(wrapped.get_historical_neighbors(nodes, times, 32),
                    sampler._historical_neighbors_reference(nodes, np.minimum(times, 24.), 32)):
        np.testing.assert_array_equal(a, b)
    for a in sampler.get_historical_neighbors(np.array([], dtype=int), np.array([]), 32):
        assert a.shape == (0, 32)


def test_all_empty():
    sampler = NeighborSampler([[], []], sample_neighbor_strategy='recent')
    for a in sampler.get_historical_neighbors(np.array([0, 1]), np.array([0., 1.]), 32):
        assert not a.any()
