import numpy as np
import matplotlib.pyplot as plt
from geometry.mesh_geometry import Rect, Mesh
from geometry.combined_cube_mesh import CombinedCubeMesh
from geometry.stats import Stats

plt.style.use("seaborn-v0_8")

# Constants
STEP_SIZE = 0.1

# Mesh Creating
s1 = Rect(1, 2, step_size=STEP_SIZE)
mesh = Mesh(s1)

s2 = Rect(0.5, 1.5, step_size=STEP_SIZE)
s2.transform_square([-0.5, 0.25], np.pi / 4)

# Distance Tests
strict_mesh = CombinedCubeMesh([s1, s2])
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
strict_mesh.visualize(ax, verbose=0)
plt.show()
plt.close('all')

all_points = np.vstack([
    strict_mesh.global_boundary_points,
    strict_mesh.global_inner_points,
    np.vstack([dynamic_region.filled_points for dynamic_region in strict_mesh.dynamic_regions]),
])

fig, ax = plt.subplots(1,1)
stats = Stats(all_points, strict_mesh.mesh)
stats.plot_delaunay(ax)
plt.show()
plt.close('all')

fig, ax = plt.subplots(1,1)
stats.plot_dists_pdf(ax)
plt.show()