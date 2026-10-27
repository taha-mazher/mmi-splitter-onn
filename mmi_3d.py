"""3D check of the 2D effective-index MMI (mmi_test.py): same layout as a 220 nm silicon film in oxide, TE0 at 1550 nm.
Uses the y and z mirror symmetries (TE0 is odd in y and even in z in Meep's convention), so a quarter of the cell is
simulated; the second output equals the first by symmetry. Usage: python mmi_3d.py [L_mmi_um] [resolution]"""
import sys
import time

import meep as mp

L = float(sys.argv[1]) if len(sys.argv) > 1 else 7.5
res = int(sys.argv[2]) if len(sys.argv) > 2 else 20
w, W, arm, t = 0.5, 3.0, 4.0, 0.22           # waveguide width, MMI width, access length, film thickness (um)
Si, SiO2 = mp.Medium(index=3.48), mp.Medium(index=1.44)
dpml, pad, padz = 1.0, 1.0, 1.0
sx, sy, sz = 2 * arm + L + 2 * dpml, W + 2 * pad + 2 * dpml, t + 2 * padz + 2 * dpml
fcen, df = 1 / 1.55, 0.1 / 1.55
y_out = W / 4

geom = [mp.Block(mp.Vector3(arm + dpml, w, t), center=mp.Vector3(-sx / 2 + (arm + dpml) / 2), material=Si),
        mp.Block(mp.Vector3(L, W, t), center=mp.Vector3(), material=Si)]
for y in (y_out, -y_out):
    geom.append(mp.Block(mp.Vector3(arm + dpml, w, t), center=mp.Vector3(sx / 2 - (arm + dpml) / 2, y), material=Si))

x_src, x_in, x_out = -sx / 2 + dpml + 0.5, -sx / 2 + dpml + 1.5, sx / 2 - dpml - 0.5
ms = mp.Vector3(0, 2 * w + 1, 2 * padz)
src = [mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(x_src), size=ms,
                          eig_band=1, eig_parity=mp.ODD_Y + mp.EVEN_Z)]
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), boundary_layers=[mp.PML(dpml)], geometry=geom, sources=src,
                    resolution=res, default_material=SiO2,
                    symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=+1)])
m_in = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(center=mp.Vector3(x_in), size=ms))
m_1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(center=mp.Vector3(x_out, y_out), size=ms))

t0 = time.time()
mp.verbosity(0)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(x_out, y_out), 1e-5))
wall = time.time() - t0

a_in = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=mp.ODD_Y + mp.EVEN_Z).alpha[0, 0]
a_1 = sim.get_eigenmode_coefficients(m_1, [1], eig_parity=mp.EVEN_Z).alpha[0, 0]
P_in = abs(a_in[0]) ** 2
T1, R = abs(a_1[0]) ** 2 / P_in, abs(a_in[1]) ** 2 / P_in
print(f'3D L={L} um res={res}: T1={T1:.4f} T1+T2={2 * T1:.4f} R={R:.4f} wall={wall:.0f} s', flush=True)
assert 2 * T1 + R < 1.02, 'more power out than in: source or monitor set up wrong'
