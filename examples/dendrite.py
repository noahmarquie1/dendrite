import polars as pl
from geometry.mesh_geometry import Mesh, Rect
from shapely.plotting import plot_polygon
from shapely import Polygon
import matplotlib.pyplot as plt

STEP_SIZE = 0.00004

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

for region in mesh.dynamic_regions:
    region.fill()

fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)

for region in mesh.dynamic_regions:
    plt.scatter(region.filled_points[:, 0], region.filled_points[:, 1], c="green", s=4)

plt.show()