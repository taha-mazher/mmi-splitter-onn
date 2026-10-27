"""2D (effective index, resolution 20) vs 3D (220 nm SOI, resolution 30) transmission at 1550 nm on the designs simulated
at both fidelities: error, rank agreement and where the 2D model puts the best 3D designs.
Usage: python compare.py     (reads data/lf2d_*.csv and data/hf3d_*.csv)"""
import glob

import pandas as pd

read = lambda tag: pd.concat([pd.read_csv(f) for f in glob.glob(f'data/{tag}_*.csv')]).drop_duplicates('id').set_index('id')
lf, hf = read('lf2d'), read('hf3d')
bad = lf.T1550 + lf.R1550 > 1.02  # more power out than in: the resolution-20 grid is too coarse for that design
ok = lf[~bad]
print(f'2D: {len(lf)} designs, {int(lf.ok.sum())} buildable, {int(bad.sum())} flagged non-physical (T + R > 1.02); '
      f'best physical T1550 {ok.T1550.max():.3f} (id {ok.T1550.idxmax()})')
print(f'3D: {len(hf)} designs, {int(hf.ok.sum())} buildable\n')
j = hf[hf.ok == 1][['W', 'L', 'wt', 'Lt', 's', 'T1550', 'T1530', 'T1565', 'R1550']].join(lf['T1550'], rsuffix='_2d')
j['err'] = j.T1550 - j.T1550_2d
j['rank_2d'] = j.T1550_2d.rank(ascending=False).astype(int)
print(j.sort_values('T1550', ascending=False).round(3).to_string())
print(f'\nmean |3D - 2D| {j.err.abs().mean():.3f}, mean 3D - 2D {j.err.mean():+.3f}, max |err| {j.err.abs().max():.3f}')
print(f'Spearman rank correlation {j.T1550.corr(j.T1550_2d, method="spearman"):.2f}')
top = lambda c: set(j[c].nlargest(5).index)
print(f'top-5 overlap {len(top("T1550") & top("T1550_2d"))}/5; best 3D design (id {j.T1550.idxmax()}) is 2D rank '
      f'{j.rank_2d[j.T1550.idxmax()]} of {len(j)}')
