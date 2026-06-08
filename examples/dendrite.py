import polars as pl
from geometry.mesh_geometry import Mesh, Rect
from shapely.plotting import plot_polygon
from shapely import Polygon
import matplotlib.pyplot as plt

STEP_SIZE = 0.0001

data = pl.read_csv("dendrite.csv")

base_rect = Rect(data[0]['width'].item(), data[0]['height'].item(), step_size=STEP_SIZE)
mesh = Mesh(base_rect)

for i in range(1, 3):
    print(f"Adding square {i+1}/{data.shape[0]}")

    square_data = data[i]
    rect = Rect(square_data['width'].item(), square_data['height'].item(), step_size=STEP_SIZE)
    rect.transform_square(
        offset=[square_data['trans_x'].item(), square_data['trans_y'].item()], 
        theta=square_data['rotation'].item()
    )
    mesh.add_rect(rect)



mesh.dynamic_fill(verbose=1)
mesh.dynamic_regions[0].visualize()
plt.show()


plt.close('all')
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)
for region in mesh.static_regions.values():
    region.visualize(ax)

region = mesh.intersections[base_rect][0]
plt.scatter(region.filled_points[:, 0], region.filled_points[:, 1], c="green", s=4)

plt.scatter(mesh.boundary_points[:, 0], mesh.boundary_points[:, 1], c='red')

print(mesh.intersections[base_rect])
plt.show()