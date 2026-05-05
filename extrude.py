# should include functions to make a point cloud 3D and plot a corresponding graph
from geometry import make_line

import matplotlib.pyplot as plt
import numpy as np


def extrude(base, step_size, start_z, end_z):
    base = np.hstack([base, np.zeros((base.shape[0], 1))]) # 2d - 3d conversion
    base = base + np.array([0, 0, start_z])

    points = base
    for point in base:
        start = point
        end = point + np.array([0, 0, end_z - start_z])
        line = make_line(start, end, step_size)
        points = np.vstack([points, line])

    return points


def plot_3d_element(points, boundaries=None, ax=None, color="lightblue", alpha=0.8):
    if ax is None:
        ax = plt.gca()
    points = np.asarray(points)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, color=color, alpha=alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud")

    if boundaries is not None:
        ax.set_xlim(boundaries[0][0], boundaries[0][1])
        ax.set_ylim(boundaries[1][0], boundaries[1][1])
        ax.set_zlim(boundaries[2][0], boundaries[2][1])