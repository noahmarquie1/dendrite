import polars as pl
from geometry.mesh_geometry import Mesh, Rect, StaticRegion
from shapely.plotting import plot_polygon
from shapely import Polygon
import matplotlib.pyplot as plt

STEP_SIZE = 0.00006

data = pl.read_csv("dendrite.csv")

base_rect = Rect(data[0]['width'].item(), data[0]['height'].item(), step_size=STEP_SIZE)
mesh = Mesh(base_rect)

for i in range(1, data.shape[0]):
    print(f"Adding square {i+1}/{data.shape[0]}")

    square_data = data[i]
    rect = Rect(square_data['width'].item(), square_data['height'].item(), step_size=STEP_SIZE)
    rect.transform_square(
        offset=[square_data['trans_x'].item(), square_data['trans_y'].item()], 
        theta=square_data['rotation'].item()
    )
    mesh.add_rect(rect)

for i, region in enumerate(mesh.dynamic_regions):
    #verbose = (i == len(mesh.dynamic_regions) - 1)
    verbose = 0
    region.fill(verbose=verbose)

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)

for region in mesh.dynamic_regions:
    #plt.scatter(region.boundary_points[:, 0], region.boundary_points[:, 1], c='red', s=4, alpha=0.6)
    plt.scatter(region.filled_points[:, 0], region.filled_points[:, 1], s=4)

plt.show()