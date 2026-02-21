import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def compute_fields_vectorized(X, Y, z, xm, m):
    """
    Fully vectorized computation of U and gz over a grid.
    """
    # Flatten grid
    XY = np.column_stack([X.ravel(), Y.ravel()])
    N = XY.shape[0]

    # Build observation array (N,3)
    obs = np.column_stack([XY, np.full(N, z)])

    # Vectorized distance
    xm = np.array(xm)
    r_vec = obs - xm
    r = np.linalg.norm(r_vec, axis=1)

    G = 6.674e-11

    U = G * m / r
    gz = G * m * (obs[:, 2] - xm[2]) / (r**3)

    return U.reshape(X.shape), gz.reshape(X.shape)


def create_contour_plots(grid_spacing):
    """Create contour plots for a single mass anomaly."""

    # Mass parameters
    m = 1.0e7
    xm = np.array([0.0, 0.0, -10.0])

    # Grid
    x_vals = np.arange(-100, 100 + grid_spacing, grid_spacing)
    y_vals = np.arange(-100, 100 + grid_spacing, grid_spacing)
    X, Y = np.meshgrid(x_vals, y_vals)

    elevations = [0, 10, 100]

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

    fig.suptitle(
        f'Gravity Potential and Effect for Single Mass Anomaly\nGrid Spacing: {grid_spacing} m',
        fontsize=14
    )

    # First pass: compute all grids
    U_grids = []
    gz_grids = []

    for z in elevations:
        U_grid, gz_grid = compute_fields_vectorized(X, Y, z, xm, m)
        U_grids.append(U_grid)
        gz_grids.append(gz_grid)

    # Global limits (same for all U, same for all gz)
    U_min = min(U.min() for U in U_grids)
    U_max = max(U.max() for U in U_grids)
    gz_min = min(gz.min() for gz in gz_grids)
    gz_max = max(gz.max() for gz in gz_grids)

    # Plotting
    for i, z in enumerate(elevations):

        # ---- Potential ----
        ax1 = axes[i, 0]
        im1 = ax1.contourf(
            X, Y, U_grids[i],
            levels=50,
            cmap='viridis',
            vmin=U_min,
            vmax=U_max
        )

        # EXACT assignment specification:
        ax1.plot(X, Y, "xk", markersize=2)

        ax1.set_title(f'Gravity Potential at z = {z} m')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_aspect('equal')
        fig.colorbar(im1, ax=ax1, label='Potential (m²/s²)')

        # ---- Gravity Effect ----
        ax2 = axes[i, 1]
        im2 = ax2.contourf(
            X, Y, gz_grids[i],
            levels=50,
            cmap='plasma',
            vmin=gz_min,
            vmax=gz_max
        )

        # EXACT assignment specification:
        ax2.plot(X, Y, "xk", markersize=2)

        ax2.set_title(f'Gravity Effect at z = {z} m')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_aspect('equal')
        fig.colorbar(im2, ax=ax2, label='g_z (m/s²)')

    output_file = f'single_mass_grid_{grid_spacing:.1f}m.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    print(f"Saved plot: {output_file}")

    plt.show()


if __name__ == "__main__":
    print("Creating contour plots for single mass anomaly...")
    create_contour_plots(5.0)
    create_contour_plots(25.0)
    print("Done!")