"""Latin-hypercube dataset of MMI designs at one fidelity, appended to data/<tag>_<shard>.csv and resumable.
Knobs: W 2-4 um; r 0.6-1.4, the MMI length as a fraction of the self-imaging estimate L0 = 7.5 * (W/3)^2 (2D optimum
at W = 3); wt 0.5-1.5 um; Lt 0-3 um; s 0.18-0.32. Designs whose output tapers overlap are recorded with ok=0.
The design list depends only on N and seed, so every fidelity and shard sees the same designs; FIRST=K (env)
keeps only ids < K, so a small high-fidelity set is a subset of the low-fidelity one.
Usage (WSL): python dataset.py TAG DIM RES N SHARD NSHARDS [SEED]     e.g. python dataset.py lf2d 2 20 2000 0 2"""
import csv
import glob
import os
import sys

import numpy as np
from scipy.stats import qmc

from mmi import simulate

LO = np.array([2.0, 0.6, 0.5, 0.0, 0.18])
HI = np.array([4.0, 1.4, 1.5, 3.0, 0.32])
NAMES = ['W', 'r', 'wt', 'Lt', 's']


def designs(n, seed=0):
    x = qmc.scale(qmc.LatinHypercube(d=5, seed=seed).random(n), LO, HI)
    return [dict(id=i, **dict(zip(NAMES, np.round(v, 4)))) for i, v in enumerate(x)]


def main():
    tag, dim, res, n, shard, nshards = sys.argv[1], *map(int, sys.argv[2:7])
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'{tag}_{shard}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    done = {int(r['id']) for f in glob.glob(out.replace(f'_{shard}.csv', '_*.csv')) for r in csv.DictReader(open(f))}
    pool = os.environ.get('POOL')  # workers share every remaining design instead of a fixed slice; mkdir claims each one
    claims = os.path.join(os.path.dirname(out), 'claims')
    cols = ['id', *NAMES, 'L', 'ok', 'T1530', 'T1550', 'T1565', 'R1550', 'wall']
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, cols)
        if new:
            w.writeheader()
        todo = designs(n, seed)[:int(os.environ.get('FIRST', n))]
        for d in todo if pool else todo[shard::nshards]:
            if d['id'] in done:
                continue
            if pool:
                try:
                    os.makedirs(os.path.join(claims, f"{tag}_{d['id']}"))
                except FileExistsError:
                    continue
            d['L'] = round(d['r'] * 7.5 * (d['W'] / 3) ** 2, 4)
            res_ = simulate(d['W'], d['L'], d['wt'], d['Lt'], d['s'], dim, res)
            w.writerow(dict(d, ok=int(res_ is not None), **(res_ or {})))
            f.flush()


if __name__ == '__main__':
    main()
