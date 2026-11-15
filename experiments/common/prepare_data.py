"""Materialize the four JODIE datasets for both final experiment suites."""
from etl.prepare_benchmark import CTDG,prepare,ROOT
if __name__ == '__main__':
    for name in CTDG:prepare(name,32768,ROOT/'data/benchmarks/events32768')
