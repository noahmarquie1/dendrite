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
mesh.add_rect(s2, verbose=1)
#mesh.dynamic_fill(verbose=0)


# Distance Tests
strict_mesh = CombinedCubeMesh([s1, s2])
fig, ax = plt.subplots(1,1)
strict_mesh.visualize(ax, verbose=1)
plt.savefig("rects.png")
plt.close('all')

all_points = np.vstack([
    strict_mesh.global_boundary_points,
    strict_mesh.global_inner_points,
    np.vstack([dynamic_region.filled_points for dynamic_region in strict_mesh.dynamic_regions]),
])
stats = Stats(all_points, strict_mesh.mesh, delaunay_out="out/tri.png")

fig, ax = plt.subplots(1,1)
stats.make_dists_pdf(ax)
plt.show()


quit()
# Plotting and Animation
plt.close('all')
mesh.animate_region(mesh.dynamic_regions[0])

fig, ax = plt.subplots(1,1)
fig.set_size_inches(8, 8)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)

ax.scatter(mesh.dynamic_regions[0].filled_points[:, 0], mesh.dynamic_regions[0].filled_points[:, 1], alpha=0.5, s=10)
ax.scatter(mesh.dynamic_regions[0].connecting_points[:, 0], mesh.dynamic_regions[0].connecting_points[:, 1], alpha=0.5, s=10)
ax.scatter(mesh.boundary_points[:, 0], mesh.boundary_points[:, 1], alpha=0.5, s=10)

img_out = "rects.png"
print(f"Saving image to {img_out}.")
plt.savefig(img_out)
