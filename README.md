# MMI splitters for photonic neural networks: what a fast 2D model gets wrong

A 1x2 multimode-interference (MMI) power splitter is the building block of the Mach-Zehnder meshes used in photonic
neural networks. Every light path through an N x N Clements mesh crosses 2N splitters, so a few percent of loss per
splitter compounds into most of the light. This project designs MMI splitters with FDTD (Meep), checks how far the
usual fast 2D approximation is from a full 3D simulation, and puts the resulting splitters into a trained photonic
network to see what their loss costs in laser power.

Everything runs on a laptop (6 cores, 15 GB RAM, no GPU needed): 2D designs take about 4 s each, 3D designs about
an hour each.

## Setup

- **Device:** a 220 nm silicon film in oxide, 0.5 um access waveguides, TE0 input, C band (1530, 1550, 1565 nm).
- **Knobs:** MMI width W (2-4 um), length L (as a fraction r = 0.6-1.4 of the self-imaging estimate
  L0 = 7.5 (W/3)^2 um), taper width wt where the access guides meet the MMI (0.5-1.5 um), taper length Lt (0-3 um),
  and output offset s (ports at +-sW, s = 0.18-0.32).
- **2D model:** the film replaced by its slab effective index (2.85), resolution 20 px/um.
- **3D model:** the actual film, resolution 30 px/um, y and z mirror symmetries (a quarter of the cell).
- **Metric:** total transmission into the fundamental mode of both outputs, T = T1 + T2 (1.0 = lossless).

## Results

### 1. The plain MMI: 2D says 90%, 3D says 77%

For an untapered 3 um wide MMI, the 2D length sweep peaks at L = 7.5 um with T = 0.895, and the value is converged
(0.894 to 0.897 from resolution 20 to 60). The same device in 3D:

| L (um) | 2D, res 30 | 3D, res 20 | 3D, res 30 |
|---|---|---|---|
| 7.5 | 0.895 | 0.662 | 0.641 |
| 8.0 | 0.872 | 0.810 | 0.766 |

2D overestimates the transmission by 11 to 25 points and puts the optimum at the wrong length.
[`logs/2d_length_sweep_and_resolution.log`](logs/2d_length_sweep_and_resolution.log),
[`logs/3d_untapered_check.log`](logs/3d_untapered_check.log)

### 2. Tapered designs: 2D cannot be trusted to rank them

2,000 Latin-hypercube designs were simulated in 2D (1,423 buildable; the rest have overlapping output tapers). The
first 36 were rerun in 3D, of which 23 are buildable. On those 23 ([`results/compare_2d_3d.txt`](results/compare_2d_3d.txt)):

- mean |3D - 2D| = 0.111, with errors in both directions (from -0.18 to +0.60), so it is not a fixed offset
  that a correction factor could remove;
- Spearman rank correlation 0.68; 4 of the 2D top five are in the 3D top five;
- **the best design in 3D (T = 0.922; W 3.39, L 9.07, wt 1.36, Lt 2.37 um) is ranked 20th of 23 by 2D**, which
  predicts 0.32 for it. Optimising on the 2D model alone would have discarded it.

Tapers help: the best tapered design reaches 0.922 in 3D against 0.766 for the best of the untapered lengths checked.

### 3. What splitter loss costs a photonic neural network

A 16 x 16 Clements mesh (120 MZIs, 32 splitters on every path) was trained with ideal splitters on handwritten digits
(sklearn 8 x 8 digits, PCA to 16 features, 10 classes; 91.9% test accuracy), then tested with lossy splitters and
1 uW rms photodetector noise per port ([`results/onn_exp1.txt`](results/onn_exp1.txt)):

| Splitter transmission | Light left after 32 splitters | Accuracy at 0.1 mW | at 1 mW | at 10 mW |
|---|---|---|---|---|
| ideal (1.0) | 100% | 91.9% | 91.9% | 91.9% |
| commercial-grade (0.98) | 52% | 90.8% | 91.8% | 91.9% |
| best 3D-verified tapered MMI (0.922) | 7.4% | 63.6% | 88.9% | 89.3% |
| untapered MMI, 2D estimate (0.898) | 3.2% | 37.0% | 85.7% | 87.3% |
| untapered MMI, 3D (0.766) | 0.02% | 9.8% | 13.4% | 32.9% |

Loss that is equal on every path only dims the output, so it is paid for in laser power: the best splitter here
needs roughly ten times the laser power of a commercial-grade one for the same accuracy, and the untapered 3D design
does not work at any power tried. A designer trusting the 2D estimate of the untapered MMI (0.898) would expect a
working network and get one near chance level.

Static fabrication errors, at ideal loss (mean of 10 simulated chips): a phase error of 0.05 rad rms costs 1.7 points
and 0.1 rad costs 8; a splitting-ratio error of 0.05 rms costs 5 points and 0.1 costs 21. The mesh is far more
sensitive to splitting ratio than to phase.

## What did not hold up

- **My first explanation of the 3D shortfall was wrong.** I first put the dip at L = 7.5 um down to the coarse 3D
  grid. Rerunning at resolution 30 kept it (0.662 to 0.641), so it is real physics that the effective-index model
  misses, not a numerical artefact.
- **The best 2D designs were never checked in 3D.** The 2D set has designs predicted at up to 0.997, but only the first
  36 designs (a random subset) went through 3D, because each takes about an hour. Given the errors above, those 2D
  numbers should not be read as achievable.
- **One 2D design returned more power than it was given** (T + R of about 1.2, design 741), a resolution-20 artefact. It is
  flagged and excluded from the 2D best.

## Limitations

- 3D at resolution 30 is not fully converged: going from 20 to 30 moved T by 2 to 4 points on the untapered checks.
- Only 23 designs have both fidelities, so the error statistics are rough.
- The network model treats every splitter as identical and only includes detector noise and static errors; it ignores
  crosstalk, thermal drift and wavelength dependence. The digits task is small.
- No fabricated device: everything here is simulation.

## Reproduce

Meep 1.35 (no MPI) and SciPy in WSL for the FDTD; PyTorch and scikit-learn for the network (CPU is enough).

```bash
python mmi_test.py 7.5 30 2.85                # one 2D untapered MMI: L, resolution, core index
python mmi_3d.py 8.0 30                       # the same device in 3D (about 30 min)
python mmi.py 3.39 9.07 1.36 2.37 0.25 3 30   # any design: W L wt Lt s dim resolution
bash run_lf2d.sh                              # 2,000 2D designs (about 1 h on 2 cores)
bash run_hf3d.sh                              # the first 36 in 3D (7-8 h on 3 cores; memory guard included)
python compare.py                             # 2D vs 3D table
cd onn && python exp1_splitters.py            # photonic network experiment (about 2 min)
```

`dataset.py` is resumable and can split the designs across workers either in fixed shards or, with `POOL=1`, as a
shared queue where each worker claims the next unfinished design.
