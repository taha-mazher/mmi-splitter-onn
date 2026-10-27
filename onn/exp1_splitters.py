"""Experiment 1: what our simulated MMI splitters do to a photonic neural network.

A 16x16 Clements mesh (120 MZIs; every light path crosses 16 MZIs = 32 splitters) is trained with ideal splitters on
handwritten digits (sklearn 8x8 digits, PCA to 16 features, 10 classes read on ports 0-9). It is then tested with
  ideal (t = 1), good commercial (t = 0.98), the best tapered MMI of the 3D set (0.922, id 10), our 2D MMI (0.898)
  and our untapered 3D MMI (0.766, 3D res 30, L = 8 um),
at laser powers from 1 uW to 10 mW per inference, with photodetector noise of 1 uW rms per port.
Equal loss on every path only dims the output, so it costs signal-to-noise (laser power), not the matrix itself.
Then: static phase and splitting-ratio errors at ideal loss, which do change the matrix.
Usage: python exp1_splitters.py   (about 2 minutes on CPU)"""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from onn import Mesh

N, CLASSES, NOISE_W = 16, 10, 1e-6
SPLITTERS = {'ideal': 1.0, 'commercial 0.98': 0.98, 'best 3D-verified MMI 0.922': 0.922,
             'our 2D MMI 0.898': 0.898, 'our 3D MMI 0.766': 0.766}


def data():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    p = PCA(N, random_state=0).fit(Xtr)
    enc = lambda A: torch.tensor((lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True))(p.transform(A)),
                                 dtype=torch.complex128)  # unit total optical power per sample
    return enc(Xtr), torch.tensor(ytr, dtype=torch.long), enc(Xte), torch.tensor(yte, dtype=torch.long)


def train(mesh, X, y, epochs=300):
    scale = torch.nn.Parameter(torch.tensor(10.0))
    opt = torch.optim.Adam([*mesh.parameters(), scale], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        p = mesh(X).abs() ** 2
        loss = torch.nn.functional.cross_entropy(scale * p[:, :CLASSES], y)
        loss.backward()
        opt.step()
    return loss.item()


def accuracy(mesh, X, y, P=None, rng=None, **err):
    with torch.no_grad():
        p = mesh(X, rng=rng, **err).abs().numpy() ** 2
    if P is not None:  # laser power P (W) per inference; photodetector noise NOISE_W rms per port
        p = P * p + NOISE_W * rng.standard_normal(p.shape)
    return float((p[:, :CLASSES].argmax(1) == y.numpy()).mean())


def main():
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = data()
    mesh = Mesh(N)
    print(f'trained with ideal splitters, final loss {train(mesh, Xtr, ytr):.3f}; '
          f'clean test accuracy {accuracy(mesh, Xte, yte):.3f}')
    powers = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    print('\naccuracy vs laser power per inference (detector noise 1 uW rms):')
    print(f"{'splitter':28s}" + ''.join(f'{p * 1e3:>9.3f}mW' for p in powers) + '   light left after 32 splitters')
    for name, t in SPLITTERS.items():
        rng = np.random.default_rng(0)
        accs = [np.mean([accuracy(mesh, Xte, yte, P, rng, t=t) for _ in range(5)]) for P in powers]
        print(f'{name:28s}' + ''.join(f'{a:11.3f}' for a in accs) + f'   {t ** 32:.2e}')
    print('\nstatic fabrication errors, ideal loss (mean of 10 chips):')
    for kind, vals in (('phase_sigma', [0.01, 0.05, 0.1, 0.2]), ('eta_sigma', [0.01, 0.02, 0.05, 0.1])):
        rng = np.random.default_rng(1)
        accs = [np.mean([accuracy(mesh, Xte, yte, rng=rng, **{kind: v}) for _ in range(10)]) for v in vals]
        print(f'{kind:12s} ' + '  '.join(f'{v}: {a:.3f}' for v, a in zip(vals, accs)))


if __name__ == '__main__':
    main()
