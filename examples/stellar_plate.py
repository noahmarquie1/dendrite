import polars as pl
import numpy as np
from geometry.hex_geometry import Hexagon
from geometry.strict_mesh import StrictMesh
import matplotlib.pyplot as plt
from geometry.stats import Stats

# Constants
STEP_SIZE = 0.0002

# Generate Stellar Plate
data = pl.read_csv("data/stellar_plate.csv")
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(8)

ax.set_aspect(1)

hex_list: list[Hexagon] = []
for row in data.iter_rows():
    radius, trans_x, trans_y, theta = row
    hex = Hexagon(radius, STEP_SIZE)
    hex.transform((trans_x, trans_y), theta)
    hex_list.append(hex)


mesh = StrictMesh(hex_list, dynamic=True)

# Visualize
plot_out = "out/plot.png"
mesh.visualize(ax)
plt.savefig(plot_out)
print(f"Plot saved to {plot_out}.")
plt.close('all')

all_points = np.vstack([
    mesh.global_boundary_points,
    mesh.global_inner_points,
    np.vstack([dynamic_region.filled_points for dynamic_region in mesh.dynamic_regions]),
])
stats = Stats(all_points, mesh.mesh, buffer=STEP_SIZE*0.01)

fig, ax = plt.subplots(1,1)
delaunay_out = "out/tri.png"
fig.set_figwidth(10)
fig.set_figheight(20)
stats.plot_delaunay(ax=ax)
plt.savefig(delaunay_out)
print(f"Delaunay Triangulation saved to {delaunay_out}.")
plt.close('all')

fig, ax = plt.subplots(1,1)
pdf_out = "out/pdf.png"
stats.plot_dists_pdf(ax)
plt.savefig(pdf_out)
print(f"PDF saved to {pdf_out}.")
plt.close('all')

