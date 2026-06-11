import polars as pl
from geometry.mesh_geometry import Rect
from geometry.combined_cube_mesh import CombinedCubeMesh
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

STEP_SIZE = 0.00005

# Generate geometry from dendrite.csv
data = pl.read_csv("dendrite.csv")

base_rect = Rect(data[1]['width'].item(), data[1]['height'].item(), step_size=STEP_SIZE)
base_rect.transform_square(
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
    print(f"Adding square {i+1}/{len(indices)}")

    square_data = data[index]
    rect = Rect(square_data['width'].item(), square_data['height'].item() * height_multiples[i], step_size=STEP_SIZE)
    rect.transform_square(
        offset=[square_data['trans_x'].item() * x_offsets[i], square_data['trans_y'].item()], 
        theta=square_data['rotation'].item()
    )
    rect_list.append(rect)


mesh = CombinedCubeMesh(rect_list)