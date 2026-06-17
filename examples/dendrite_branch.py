import polars as pl
from geometry.mesh_geometry import Rect
from geometry.strict_mesh import StrictMesh
import matplotlib.pyplot as plt
import os
from geometry.stats import Stats
from shapely.plotting import plot_polygon
import numpy as np

STEP_SIZE = 0.00005

# Generate geometry from dendrite.csv
data = pl.read_csv("data/dendrite.csv")

base_rect = Rect(data[1]['width'].item(), data[1]['height'].item(), step_size=STEP_SIZE)
base_rect.transform(
    offset=[data[1]['trans_x'].item(), data[1]['trans_y'].item()], 
    theta=data[1]['rotation'].item()
)
rect_list = [base_rect]

indices = [7,8,19,20,31,32,43,44,55,56,67,68]
x_offsets = [0.7, 0.7, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7]
height_multiples = [2, 2, 1.3, 1.3, 1, 1, 1, 1, 1.2, 1.2, 2, 2]

fig, ax = plt.subplots(1,1)
ax.set_aspect(1)

for i, index in enumerate(indices):
    #print(f"Adding square {i+1}/{len(indices)}")

    square_data = data[index]
    rect = Rect(square_data['width'].item(), square_data['height'].item() * height_multiples[i], step_size=STEP_SIZE)
    rect.transform(
        offset=[square_data['trans_x'].item() * x_offsets[i], square_data['trans_y'].item()], 
        theta=square_data['rotation'].item()
    )
    rect_list.append(rect)


plt.style.use("seaborn-v0_8")
mesh = StrictMesh(rect_list, dynamic=True)
out_dir = "out/"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(1,1)
fig.set_figwidth(10)
fig.set_figheight(20)
ax.set_aspect(1)

img_out = "out/plot.png"
mesh.visualize(ax=ax)
plt.savefig(img_out)
plt.close('all')
print(f"Plot saved to {img_out}.")

# Save stats
fig, ax = plt.subplots(1,1)
pdf_out = "out/pdf.png"

all_points = np.vstack([
    mesh.global_boundary_points,
    mesh.global_inner_points,
    np.vstack([dynamic_region.filled_points for dynamic_region in mesh.dynamic_regions]),
])
stats = Stats(all_points, mesh.mesh, buffer=STEP_SIZE*0.01)

delaunay_out = "out/tri.png"
fig.set_figwidth(10)
fig.set_figheight(20)
stats.plot_delaunay(ax=ax)
plt.savefig(delaunay_out)
print(f"Delaunay Triangulation saved to {delaunay_out}.")
plt.close('all')

fig, ax = plt.subplots(1,1)
stats.plot_dists_pdf(ax)
plt.savefig(pdf_out)
print(f"PDF saved to {pdf_out}")
