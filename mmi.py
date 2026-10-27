"""Parametric 1x2 MMI power splitter in Meep, 2D (effective index) or 3D (220 nm silicon film), TE0 at 1530/1550/1565 nm.

Knobs (um): W  MMI width, L  MMI length, wt  taper width where it meets the MMI, Lt  taper length (0 = no taper),
s  output offset as a fraction of W (ports at +-s*W). Access waveguides are 0.5 um wide.
Returns the power fraction in each output's fundamental mode (T1, T2) and the reflection into the input's,
at each of the three wavelengths. 2D uses the slab effective index 2.85; 3D uses the y and z mirror symmetries.
Usage: python mmi.py W L wt Lt s [dim] [res]     e.g. python mmi.py 3 7.5 0.5 0 0.25 2 20"""
import sys
import time

import meep as mp

W0 = 0.5                      # access waveguide width
WL = (1.53, 1.55, 1.565)      # C-band edges and centre
N_EFF2D, N_SI, N_OX, T_SI = 2.85, 3.48, 1.44, 0.22
ARM, PAD, PADZ, DPML = 3.0, 1.0, 1.0, 1.0


def geometry(W, L, wt, Lt, s, core, height):
    """Input guide + taper, MMI body, two output tapers + guides, along x; centred on the MMI."""
    xin = -L / 2 - Lt
    blk = lambda cx, cy, sx, sy: mp.Block(mp.Vector3(sx, sy, height), center=mp.Vector3(cx, cy), material=core)
    taper = lambda x0, x1, y, w0, w1: mp.Prism([mp.Vector3(x0, y - w0 / 2), mp.Vector3(x1, y - w1 / 2),
                                                mp.Vector3(x1, y + w1 / 2), mp.Vector3(x0, y + w0 / 2)],
                                               height=height, center=mp.Vector3((x0 + x1) / 2, y), material=core)
    far = 50.0  # guides run into the PML
    g = [blk(xin - far / 2, 0, far, W0), blk(0, 0, L, W)]
    for y in (s * W, -s * W):
        g.append(blk(L / 2 + Lt + far / 2, y, far, W0))
    if Lt > 0:
        g.append(taper(xin, -L / 2, 0, W0, wt))
        g += [taper(L / 2, L / 2 + Lt, y, wt, W0) for y in (s * W, -s * W)]
    return g


def simulate(W, L, wt, Lt, s, dim=2, res=20):
    if 2 * s * W - wt < 0.1 or s * W + wt / 2 > W / 2:  # output tapers overlap or stick out of the MMI
        return None
    three = dim == 3
    core = mp.Medium(index=N_SI if three else N_EFF2D)
    sx, sy = L + 2 * (Lt + ARM) + 2 * DPML, W + 2 * PAD + 2 * DPML
    sz = T_SI + 2 * PADZ + 2 * DPML if three else 0
    freqs = [1 / w for w in WL]
    fcen, df = 1 / 1.55, 0.1 / 1.55
    ms = mp.Vector3(0, 2 * W0 + 1, 2 * PADZ if three else 0)
    x_src, x_in, x_out = -sx / 2 + DPML + 0.5, -sx / 2 + DPML + 1.5, sx / 2 - DPML - 0.5
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz), boundary_layers=[mp.PML(DPML)], resolution=res,
        geometry=geometry(W, L, wt, Lt, s, core, T_SI if three else mp.inf), default_material=mp.Medium(index=N_OX),
        sources=[mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(x_src), size=ms,
                                    eig_band=1, eig_parity=mp.ODD_Y + mp.EVEN_Z)],  # TE0 (Meep: odd in y)
        symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=+1)] if three else [])
    m_in = sim.add_mode_monitor(freqs, mp.ModeRegion(center=mp.Vector3(x_in), size=ms))
    m_out = sim.add_mode_monitor(freqs, mp.ModeRegion(center=mp.Vector3(x_out, s * W), size=ms))
    t0 = time.time()
    mp.verbosity(0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey if three else mp.Hz,
                                                           mp.Vector3(x_out, s * W), 1e-5 if three else 1e-6))
    a_in = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=mp.ODD_Y + mp.EVEN_Z).alpha[0]
    a_out = sim.get_eigenmode_coefficients(m_out, [1], eig_parity=mp.EVEN_Z).alpha[0]
    p_in = abs(a_in[:, 0]) ** 2
    T = abs(a_out[:, 0]) ** 2 / p_in            # each output; the two are equal by symmetry
    R = abs(a_in[:, 1]) ** 2 / p_in
    if (2 * T + R > 1.02).any():  # coarse grids can overshoot by a few points; keep the values, the analysis flags them
        print(f'warning: more power out than in ({(2 * T + R).max():.3f}), grid too coarse for this design', flush=True)
    return dict(T1530=2 * T[0], T1550=2 * T[1], T1565=2 * T[2], R1550=R[1], wall=time.time() - t0)


if __name__ == '__main__':
    a = sys.argv[1:]
    print(simulate(*map(float, a[:5]), *(int(x) for x in a[5:7])))
