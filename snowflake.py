import matplotlib.pyplot as plt
import numpy as np
from geometry import make_square_edges, fill_in_square, remove_in_area_points
from stats import plot_mesh_pdf

STEP_SIZE = 0.1

# Define base transformations
tl_transform = np.array([
    [np.cos(-np.pi/2 + np.arctan(1/2)), -np.sin(-np.pi/2 + np.arctan(1/2))],
    [np.sin(-np.pi/2 + np.arctan(1/2)),  np.cos(-np.pi/2 + np.arctan(1/2))]
])
tl_offset = np.array([-3/4, 1/2])

t_offset = np.array([0, 0.7])

tr_transform = np.array([
    [np.cos(np.pi/2 - np.arctan(1/2)), -np.sin(np.pi/2 - np.arctan(1/2))],
    [np.sin(np.pi/2 - np.arctan(1/2)),  np.cos(np.pi/2 - np.arctan(1/2))]
])
tr_offset = np.array([3/4, 1/2])

br_transform = np.array([
    [np.cos(-np.pi/2 - np.arctan(1/2)), -np.sin(-np.pi/2 - np.arctan(1/2))],
    [np.sin(-np.pi/2 - np.arctan(1/2)),  np.cos(-np.pi/2 - np.arctan(1/2))]
])
br_offset = np.array([3/4, -1/2])

b_transform = np.array([
    [-1, 0],
    [0, -1]
])
b_offset = np.array([0, -0.7])

bl_transform = np.array([
    [np.cos(-np.pi/2 - np.arctan(1/2)), -np.sin(-np.pi/2 - np.arctan(1/2))],
    [np.sin(-np.pi/2 - np.arctan(1/2)),  np.cos(-np.pi/2 - np.arctan(1/2))]
])
bl_offset = np.array([-3/4, -1/2])

# creates one snowflake branch with no root rotation
def create_snowflake_branch(width, length, n, step_size):
    square = np.array([ # basic box for snowflake partition
        [-width / 2, 0], [width / 2, 0],
        [width / 2, length], [-width / 2, length]
    ])
    points = make_square_edges(square, step_size)
    points = np.append(points, fill_in_square(square, step_size), axis=0)

    if n > 0:
        offset = np.array([0, length / 2])
        left_branch_points = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)
        right_branch_points = create_snowflake_branch(width / 2, length / 2, n - 1, step_size)
        l_transform = np.array([
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ])

        r_transform = np.array([
            [np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
            [np.sin(-np.pi / 4), np.cos(-np.pi / 4)],
        ])

        left_branch_points = left_branch_points @ l_transform.T + offset
        left_branch_points = remove_in_area_points(left_branch_points, square)
        right_branch_points = right_branch_points @ r_transform.T + offset
        right_branch_points = remove_in_area_points(right_branch_points, square)

        points = np.append(points, left_branch_points, axis=0)
        points = np.append(points, right_branch_points, axis=0)
    return points


def create_snowflake_branch_outline(width, length, n, step_size):
    base_square = np.array([  # basic box for snowflake partition
        [-width / 2, 0], [width / 2, 0],
        [width / 2, length], [-width / 2, length]
    ])
    points = make_square_edges(base_square, step_size)

    if n > 0:
        offset = np.array([0, length / 2])
        left_branch_points, l_square = create_snowflake_branch_outline(width / 2, length / 2, n - 1, step_size)
        right_branch_points, r_square = create_snowflake_branch_outline(width / 2, length / 2, n - 1, step_size)
        l_transform = np.array([
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ])

        r_transform = np.array([
            [np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
            [np.sin(-np.pi / 4), np.cos(-np.pi / 4)],
        ])

        left_branch_points = left_branch_points @ l_transform.T + offset
        left_branch_points = remove_in_area_points(left_branch_points, base_square)
        right_branch_points = right_branch_points @ r_transform.T + offset
        right_branch_points = remove_in_area_points(right_branch_points, base_square)

        l_square = l_square @ l_transform.T + offset
        r_square = r_square @ r_transform.T + offset

        points = remove_in_area_points(points, l_square)
        points = remove_in_area_points(points, r_square)

        points = np.append(points, left_branch_points, axis=0)
        points = np.append(points, right_branch_points, axis=0)
    return points, base_square


def create_hexagonal_base(side_length):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    vertices = np.column_stack((side_length * np.cos(angles), side_length * np.sin(angles)))
    return vertices


def create_square_base(hexagon):
    p1 = np.array([max(hexagon[:, 0]), max(hexagon[:, 1])])
    p2 = np.array([max(hexagon[:, 0]), min(hexagon[:, 1])])
    p3 = np.array([min(hexagon[:, 0]), min(hexagon[:, 1])])
    p4 = np.array([min(hexagon[:, 0]), max(hexagon[:, 1])])
    return np.concat([[p1], [p2], [p3], [p4]], axis=0)


def plot_hexagonal_branches(branch, side_length):
    # Plot all branches with corresponding transformations
    # - (tl, t, tr, br, b, bl)
    # WARNING: Side length not yet incorporated
    points = np.array([]).reshape(0,2)

    branch_tl = branch @ tl_transform + tl_offset
    points = np.append(points, branch_tl, axis=0)

    branch_t = branch + t_offset
    points = np.append(points, branch_t, axis=0)

    branch_tr = branch @ tr_transform + tr_offset
    points = np.append(points, branch_tr, axis=0)

    branch_br = branch @ br_transform.T + br_offset
    points = np.append(points, branch_br, axis=0)

    branch_b = branch @ b_transform.T + b_offset
    points = np.append(points, branch_b, axis=0)

    branch_bl = branch @ bl_transform + bl_offset
    points = np.append(points, branch_bl, axis=0)

    return points


def generate_full_snowflake(branch_length, n, step_size, outline=False):
    hexagon = create_hexagonal_base(1)
    base = create_square_base(hexagon)
    if outline:
        branch, _ = create_snowflake_branch_outline(1, branch_length, n, step_size)
    else:
        branch = create_snowflake_branch(1, branch_length, n, step_size)

    branches = plot_hexagonal_branches(branch, 1)

    branches = remove_in_area_points(branches, base)
    base = fill_in_square(base, step_size)
    if outline:
        return branches
    else:
        return np.vstack([base, branches])

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    snowflake_mesh = generate_full_snowflake(5, 3, STEP_SIZE, outline=False)
    ax[0].scatter(snowflake_mesh[:, 0], snowflake_mesh[:, 1], s=1)
    ax[0].set_title("Snowflake Point Cloud")

    plot_mesh_pdf(snowflake_mesh, STEP_SIZE, ax=ax[1])
    ax[1].set_title("Snowflake PDF")

    plt.tight_layout()
    plt.show()