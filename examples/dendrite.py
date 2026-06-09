import polars as pl
from geometry.mesh_geometry import Mesh, Rect, StaticRegion
from shapely.plotting import plot_polygon
from shapely import Polygon
import matplotlib.pyplot as plt

STEP_SIZE = 0.0001

data = pl.read_csv("dendrite.csv")

base_rect = Rect(data[1]['width'].item(), data[1]['height'].item(), step_size=STEP_SIZE)
base_rect.transform_square(
        offset=[data[1]['trans_x'].item(), data[1]['trans_y'].item()], 
        theta=data[1]['rotation'].item()
    )
mesh = Mesh(base_rect)

indices = [7,8,19,20,31,32,43,44,55,56,67,68]

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
    print(f"Filling region {i+1}/{len(mesh.dynamic_regions)}")
    #verbose = (i == len(mesh.dynamic_regions) - 1)
    verbose = 1
    region.fill(verbose=0)
    if i == -1:
        mesh.animate_region(region, debug=True)

fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)

for region in mesh.dynamic_regions:
    plt.scatter(region.boundary_points[:, 0], region.boundary_points[:, 1], s=8, alpha=0.7)
    plt.scatter(region.filled_points[:, 0], region.filled_points[:, 1], s=8, alpha=0.7)

plt.show()