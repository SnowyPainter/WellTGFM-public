"""Hash the local implementations actually used; no network dependency."""
from pathlib import Path
from experiments.target_adaptation.runtime import ROOT, digest

def local_implementations():
    paths = ['models/baselines/graphmixer/model.py', 'models/baselines/ctdg/tgn.py',
             'models/baselines/ctdg/modules.py', 'models/baselines/ctdg/sampling.py']
    return {p: digest(ROOT / p) for p in paths}
