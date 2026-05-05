import matplotlib.pyplot as plt
from shapely.constructive import constrained_delaunay_triangles

from mesh import *
from geometry import make_square_edges, fill_in_square, remove_in_area_points
from shapely import affinity
from extrude import extrude, plot_3d_element

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


if __name__ == "__main__":
    _, ax = plt.subplots(2, 1, figsize=(5, 10))
    snowflake_mesh = generate_full_snowflake(1, 3, 2, STEP_SIZE)
    snowflake_mesh.visualize(ax[0])
    ax[0].set_title("Hexagonal Snowflake Mesh")

    plot_mesh_pdf(snowflake_mesh.inner_points, STEP_SIZE, ax=ax[1])
    ax[1].set_title("PDF of Inner Points")
    plt.tight_layout()
    plt.show()

    # plot 3d visualization
    start_z = -0.2
    end_z = 0.2

    walls = extrude(snowflake_mesh.edge_points, 0.1, start_z, end_z)
    face = np.hstack([snowflake_mesh.inner_points, np.zeros((snowflake_mesh.inner_points.shape[0], 1))])
    top_face = face + np.array([0, 0, end_z])
    bottom_face = face + np.array([0, 0, start_z])

    plot_width = 6
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_element(walls, boundaries=[(-plot_width, plot_width), (-plot_width, plot_width), (start_z*4, end_z*4)], ax=ax1)
    ax1.set_title("Walls")

    ax2 = fig.add_subplot(122, projection='3d')
    points = np.vstack([top_face, bottom_face])
    plot_3d_element(points, boundaries=[(-plot_width, plot_width), (-plot_width, plot_width), (start_z*4, end_z*4)], ax=ax2)
    ax2.set_title("Faces")

    plt.show()