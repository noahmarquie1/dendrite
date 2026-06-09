import numpy as np
import matplotlib.pyplot as plt
from geometry.mesh_geometry import Rect, Mesh

# Constants
STEP_SIZE = 0.05


# Testing
s1 = Rect(1, 2, step_size=STEP_SIZE)
mesh = Mesh(s1)

s2 = Rect(0.5, 1.5, step_size=STEP_SIZE)
s2.transform_square([-0.5, 0.25], np.pi / 4)
mesh.add_rect(s2, verbose=1)
mesh.dynamic_fill(verbose=0)

mesh.animate_region(mesh.dynamic_regions[0])


plt.close('all')
fig, ax = plt.subplots(1,1)
fig.set_size_inches(8, 8)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)
ax.scatter(mesh.dynamic_regions[0].filled_points[:, 0], mesh.dynamic_regions[0].filled_points[:, 1], alpha=0.5, s=5)
ax.scatter(mesh.dynamic_regions[0].connecting_points[:, 0], mesh.dynamic_regions[0].connecting_points[:, 1], alpha=0.5, s=5)
ax.scatter(mesh.boundary_points[:, 0], mesh.boundary_points[:, 1])

img_out = "rects.png"
print(f"Saving image to {img_out}.")
plt.savefig(img_out)
