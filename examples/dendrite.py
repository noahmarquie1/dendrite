import polars as pl
from geometry.mesh_geometry import Mesh, Rect, StaticRegion
from shapely.plotting import plot_polygon, plot_line
from shapely import Polygon
import matplotlib.pyplot as plt
import shapely
import numpy as np
from scipy.spatial import KDTree

STEP_SIZE = 0.00005

data = pl.read_csv("dendrite.csv")

base_rect = Rect(data[1]['width'].item(), data[1]['height'].item(), step_size=STEP_SIZE)
base_rect.transform_square(
        offset=[data[1]['trans_x'].item(), data[1]['trans_y'].item()], 
        theta=data[1]['rotation'].item()
    )
mesh = Mesh(base_rect)

indices = [7,8,19,20,31,32,43,44,55,56,67,68]
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)

for i in indices:
    print(f"Adding square {i+1}/{data.shape[0]}")

    square_data = data[i]
    rect = Rect(square_data['width'].item(), square_data['height'].item(), step_size=STEP_SIZE)
    rect.transform_square(
        offset=[square_data['trans_x'].item(), square_data['trans_y'].item()], 
        theta=square_data['rotation'].item()
    )
    mesh.add_rect(rect)

for i, region in enumerate(mesh.dynamic_regions):
    region.set_n_bodies(region.n_bodies + 5)
    print(f"Filling region {i+1}/{len(mesh.dynamic_regions)}")
    region.fill(verbose=0)


# Store and visualize all points with no overlap
plt.close('all')
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)

all_points = np.zeros((0,2))
for region in mesh.static_regions.values():
    all_points = np.vstack([all_points, region.points])

boundary_tree = KDTree(mesh.boundary_points)
for region in mesh.dynamic_regions:
    # Ensure all connecting points are included but not double counted
    dists, _ = boundary_tree.query(region.connecting_points)
    mask = dists > 5e-5
    connecting_points = region.connecting_points[mask]

    #plt.scatter(region.boundary_points[:, 0], region.boundary_points[:, 1], c='blue', alpha=0.3, s=8)

    all_points = np.vstack([
        all_points, 
        region.filled_points,
        connecting_points,
    ])

# Include boundary points that are excluded from filled points
filled_point_tree = KDTree(all_points)
dists, _ = filled_point_tree.query(mesh.boundary_points)

mask = dists > 5e-5
new_bound_points = mesh.boundary_points[mask]

all_points = np.vstack([all_points, new_bound_points])

plt.scatter(all_points[:, 0], all_points[:, 1], s=8, c='blue', alpha=0.3)
plt.show()