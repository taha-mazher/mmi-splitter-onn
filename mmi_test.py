"""Feasibility test: 2D FDTD of a 1x2 MMI splitter (silicon in oxide, 1550 nm) in Meep.
Reports power into each output's fundamental mode, reflection, and wall time per simulation.
Sanity: outputs equal by symmetry, T1 + T2 + R <= 1. Usage: python mmi_test.py [L_mmi_um] [resolution] [core index: 3.48 bulk Si, ~2.85 effective index of 220 nm SOI]"""
import sys
import time

import meep as mp

L = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0   # MMI length, um
res = int(sys.argv[2]) if len(sys.argv) > 2 else 20      # pixels per um
n_core = float(sys.argv[3]) if len(sys.argv) > 3 else 3.48
w, W, taper, arm = 0.5, 3.0, 0.0, 4.0                    # waveguide width, MMI width, -, access length (um)
Si, SiO2 = mp.Medium(index=n_core), mp.Medium(index=1.44)
dpml, pad = 1.0, 1.0
sx, sy = 2 * arm + L + 2 * dpml, W + 2 * pad + 2 * dpml
fcen, df = 1 / 1.55, 0.1 / 1.55
y_out = W / 4

geom = [mp.Block(mp.Vector3(arm + dpml, w), center=mp.Vector3(-sx / 2 + (arm + dpml) / 2), material=Si),
        mp.Block(mp.Vector3(L, W), center=mp.Vector3(), material=Si)]
for y in (y_out, -y_out):
    geom.append(mp.Block(mp.Vector3(arm + dpml, w), center=mp.Vector3(sx / 2 - (arm + dpml) / 2, y), material=Si))

x_src, x_in, x_out = -sx / 2 + dpml + 0.5, -sx / 2 + dpml + 1.5, sx / 2 - dpml - 0.5
src = [mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(x_src), size=mp.Vector3(0, 2 * w + 1),
                          eig_band=1, eig_parity=mp.ODD_Y + mp.EVEN_Z)]  # TE0: E in the chip plane, as in SOI (Meep calls it odd in y)
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), boundary_layers=[mp.PML(dpml)], geometry=geom, sources=src,
                    resolution=res, default_material=SiO2)
mon = lambda x, y: sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(center=mp.Vector3(x, y), size=mp.Vector3(0, 2 * w + 1)))
m_in, m_1, m_2 = mon(x_in, 0), mon(x_out, y_out), mon(x_out, -y_out)

t0 = time.time()
mp.verbosity(0)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Hz, mp.Vector3(x_out, y_out), 1e-6))
wall = time.time() - t0

coef = lambda m: sim.get_eigenmode_coefficients(m, [1], eig_parity=mp.EVEN_Z).alpha[0, 0]
a_in = coef(m_in)
P_in = abs(a_in[0]) ** 2
T1, T2, R = abs(coef(m_1)[0]) ** 2 / P_in, abs(coef(m_2)[0]) ** 2 / P_in, abs(a_in[1]) ** 2 / P_in
print(f'n={n_core} L={L} um res={res}: T1={T1:.4f} T2={T2:.4f} T1+T2={T1 + T2:.4f} R={R:.4f} wall={wall:.1f} s')
assert T1 + T2 + R < 1.02, 'more power out than in: source or monitor set up wrong'
