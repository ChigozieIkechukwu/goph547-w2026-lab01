# examples/driver_multi_mass.py
"""
Course-consistent multi-mass driver (Code A equivalent, but fully vectorized)

Features:
- Generates 3 sets of 5 point masses with:
  * fixed total mass mtot = 1e7 kg
  * fixed center of mass = [0, 0, -10] m
  * all masses satisfy z <= -1 m
- Saves each set to: examples/mass_set_{k}.mat
- Computes gravitational potential U and vertical gravity effect gz on:
  * 25 m grid (9x9)
  * 5 m grid  (41x41)
  * elevations z = [0, 10, 100] m
- Uses fixed color limits for fair comparison across grids/sets
- Fully vectorized computation (very fast)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


# ----------------------------
# Config
# ----------------------------
MTOT = 1.0e7
TARGET_COM = np.array([0.0, 0.0, -10.0])

M_MEAN = MTOT / 5.0
M_STD = MTOT / 100.0

XSIG = np.array([20.0, 20.0, 2.0])

Z_MAX_ALLOWED = -1.0

ZP = np.array([0.0, 10.0, 100.0])

U_MIN, U_MAX = 0.0, 8.0e-5
G_MIN, G_MAX = 0.0, 7.0e-6

EXAMPLES_DIR = "examples"


# ----------------------------
# Vectorized Physics
# ----------------------------
def gravity_potential_point_vec(obs_xyz, src_xyz, m, G=6.674e-11):
    r_vec = obs_xyz[None, :, :] - src_xyz[:, None, :]
    r = np.linalg.norm(r_vec, axis=2)
    return np.sum(G * m[:, None] / r, axis=0)


def gravity_effect_point_vec(obs_xyz, src_xyz, m, G=6.674e-11):
    r_vec = obs_xyz[None, :, :] - src_xyz[:, None, :]
    r2 = np.sum(r_vec * r_vec, axis=2)
    r = np.sqrt(r2)
    dz = r_vec[:, :, 2]
    return np.sum(G * m[:, None] * dz / (r2 * r), axis=0)


# ----------------------------
# Grid
# ----------------------------
def make_grid(npts):
    x = np.linspace(-100.0, 100.0, npts)
    return np.meshgrid(x, x)


# ----------------------------
# Mass Generator (Code A constraints)
# ----------------------------
def generate_mass_set(rng):
    while True:
        m = rng.normal(M_MEAN, M_STD, size=(5,))
        if np.any(m[:4] <= 0):
            continue

        xm = np.zeros((5, 3))
        xm[:4, 0] = rng.normal(TARGET_COM[0], XSIG[0], 4)
        xm[:4, 1] = rng.normal(TARGET_COM[1], XSIG[1], 4)
        xm[:4, 2] = rng.normal(TARGET_COM[2], XSIG[2], 4)

        m5 = MTOT - np.sum(m[:4])
        if m5 <= 0:
            continue
        m[4] = m5

        for i in range(3):
            xm[4, i] = (TARGET_COM[i] * MTOT - np.dot(m[:4], xm[:4, i])) / m[4]

        if np.all(xm[:, 2] <= Z_MAX_ALLOWED):
            com = (m @ xm) / np.sum(m)
            if np.allclose(np.sum(m), MTOT) and np.allclose(com, TARGET_COM):
                return m, xm


def generate_mass_anomaly_sets(n_sets=3, seed=0):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    for k in range(n_sets):
        m, xm = generate_mass_set(rng)

        savemat(os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat"),
                {"m": m.reshape(-1, 1), "xm": xm})

        print(f"✓ Saved {EXAMPLES_DIR}/mass_set_{k}.mat")
        print(f"  Verified Total Mass: {np.sum(m):.2e} kg")
        print(f"  Verified Center of Mass: {(m @ xm) / np.sum(m)}")
        print(f"  All masses below -1m: {np.all(xm[:, 2] <= -1.0)}")


# ----------------------------
# Vectorized Field Computation
# ----------------------------
def compute_gravity_fields(X, Y, zp, m, xm):
    ny, nx = X.shape
    nz = len(zp)

    XY = np.column_stack([X.ravel(), Y.ravel()])
    N = XY.shape[0]

    U = np.zeros((N, nz))
    g = np.zeros((N, nz))

    for k, zobs in enumerate(zp):
        obs = np.column_stack([XY, np.full((N,), zobs)])
        U[:, k] = gravity_potential_point_vec(obs, xm, m)
        g[:, k] = gravity_effect_point_vec(obs, xm, m)

    return U.reshape(ny, nx, nz), g.reshape(ny, nx, nz)


# ----------------------------
# Plotting
# ----------------------------
def plot_gravity_potential_and_effect(set_idx, X25, Y25, U25, g25, X5, Y5, U5, g5):

    def plot_block(X, Y, U, g, grid_spacing):
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)

        fig.suptitle(
            f"Mass Set {set_idx} | mtot = 1.0e7 kg | Center of Mass z = -10 m\nGrid Spacing = {grid_spacing} m",
            weight="bold",
            fontsize=14,
        )

        for i, z_val in enumerate(ZP):
            axU = axes[i, 0]
            cfU = axU.contourf(
                X, Y, U[:, :, i],
                cmap="viridis_r",
                levels=np.linspace(U_MIN, U_MAX, 50),
                vmin=U_MIN, vmax=U_MAX,
            )
            fig.colorbar(cfU, ax=axU).set_label(r"U [$m^2/s^2$]")
            axU.set_title(f"Potential (U) at z = {z_val:.0f} m")
            axU.set_ylabel("y [m]")
            if grid_spacing == 25.0:
                axU.plot(X, Y, "xk", markersize=2)

            axg = axes[i, 1]
            cfg = axg.contourf(
                X, Y, g[:, :, i],
                cmap="magma",
                levels=np.linspace(G_MIN, G_MAX, 50),
                vmin=G_MIN, vmax=G_MAX,
            )
            fig.colorbar(cfg, ax=axg).set_label(r"$g_z$ [$m/s^2$]")
            axg.set_title(f"Gravity Effect ($g_z$) at z = {z_val:.0f} m")
            if grid_spacing == 25.0:
                axg.plot(X, Y, "xk", markersize=2)

        axes[2, 0].set_xlabel("x [m]")
        axes[2, 1].set_xlabel("x [m]")
        return fig

    fig25 = plot_block(X25, Y25, U25, g25, 25.0)
    fig25.savefig(os.path.join(EXAMPLES_DIR, f"multi_mass_grid_25_set_{set_idx}.png"), dpi=300)
    plt.close(fig25)

    fig5 = plot_block(X5, Y5, U5, g5, 5.0)
    fig5.savefig(os.path.join(EXAMPLES_DIR, f"multi_mass_grid_5_set_{set_idx}.png"), dpi=300)
    plt.close(fig5)


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    missing = [k for k in range(3)
               if not os.path.exists(os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat"))]

    if missing:
        print("Generating mass sets...")
        generate_mass_anomaly_sets(3, seed=0)

    X25, Y25 = make_grid(9)
    X5, Y5 = make_grid(41)

    for k in range(3):
        print(f"\n--- Processing Mass Set {k} ---")
        data = loadmat(os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat"))
        m = data["m"][:, 0]
        xm = data["xm"]

        print(f"Verified Total Mass: {np.sum(m):.2e} kg")
        print(f"Verified Center of Mass: {(m @ xm) / np.sum(m)}")
        print(f"All masses below -1m: {np.all(xm[:, 2] <= -1.0)}")

        U25, g25 = compute_gravity_fields(X25, Y25, ZP, m, xm)
        U5, g5 = compute_gravity_fields(X5, Y5, ZP, m, xm)

        plot_gravity_potential_and_effect(k, X25, Y25, U25, g25, X5, Y5, U5, g5)

    print("\n✅ All mass sets generated and plotted!")


if __name__ == "__main__":
    main()