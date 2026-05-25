import matplotlib.pyplot as plt
from mesh_generation.mesh import Mesh
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from mesh_generation.geometry import extrude, plot_3d_element
from mesh_generation.stats import plot_mesh_pdf

STEP_SIZE = 0.2

# Define base transformations
angles = np.array([
        90 - np.degrees(np.arctan(1 / 2)),   # tl
        0,                                   # t
        -90 + np.degrees(np.arctan(1 / 2)),  # tr
        -90 - np.degrees(np.arctan(1 / 2)),  # br
        180,                                 # b
        90 + np.degrees(np.arctan(1 / 2)),  # bl
    ])

offsets = np.array([
    [-3 / 4 +1e-7, 1 / 2 -1e-7],   # tl
    [0, 1],            # t
    [3 / 4 - 1e-7, 1 / 2 - 1e-7],    # tr
    [3 / 4, -1 / 2],   # br
    [0, -1],           # b
    [-3 / 4, -1 / 2],  # bl
])

# creates one snowflake branch with no root rotation
def create_snowflake_branch(width, length, n, step_size):
    square = np.array([ # basic box for snowflake partition
        [-width / 2, 0], [width / 2, 0],
        [width / 2, length], [-width / 2, length]
    ])
    mesh = Mesh(Polygon(square), step_size=step_size)

    if n > 0:
        left_branch_mesh = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)
        right_branch_mesh = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)

        left_polygon = affinity.rotate(left_branch_mesh.polygon, 45, origin=(0, 0))
        left_polygon = affinity.translate(left_polygon, xoff=0, yoff=length/2)
        right_polygon = affinity.rotate(right_branch_mesh.polygon, -45, origin=(0, 0))
        right_polygon = affinity.translate(right_polygon, xoff=0, yoff=length/2)

        mesh.add_shape(left_polygon)
        mesh.add_shape(right_polygon)

    return mesh


def generate_full_snowflake(side_length, branch_length, n, step_size):
    snowflake_mesh = Mesh(Polygon(np.array([
        [side_length/2, side_length], [side_length, 0],
        [side_length/2, -side_length], [-side_length / 2, -side_length],
        [-side_length, 0], [-side_length / 2, side_length],
    ])), step_size=step_size)
    for i in range(6):
        branch_mesh = create_snowflake_branch(side_length, branch_length, n, step_size)
        branch_polygon = branch_mesh.polygon
        branch_polygon = affinity.rotate(branch_polygon, angles[i], origin=(0, 0))
        branch_polygon = affinity.translate(branch_polygon, xoff=offsets[i, 0]*side_length, yoff=offsets[i, 1]*side_length)
        snowflake_mesh.add_shape(branch_polygon)
    return snowflake_mesh