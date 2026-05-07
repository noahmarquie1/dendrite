import numpy as np
from scipy.spatial import Delaunay, KDTree


def make_line(p1, p2, step_size):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    m = p2 - p1
    total_dist = np.linalg.norm(m)
    if total_dist < 1e-8:
        unit_step = np.zeros(shape=(0, len(p1)))
    else:
        unit_step = m / total_dist  # unit vector in direction of line

    total_steps = int(total_dist / step_size)
    if total_steps == 0:
        return np.zeros(shape=(0, len(p1)))
    actual_step_size = total_dist / total_steps
    local_step = unit_step * actual_step_size

    points = np.array([p1])
    for i in range(1, total_steps):
        points = np.append(points, [points[i - 1] + local_step], axis=0)
    return points


def make_square_edges(s, step_size):
    total_points = np.array([]).reshape(0, 2)
    s = np.append(s, [s[0]], axis=0)
    for i in range(s.shape[0] - 1):
        total_points = np.append(total_points, make_line(s[i], s[i+1], step_size), axis=0)
    return total_points


def in_area(p, s):
    tri = Delaunay(s)
    return tri.find_simplex(p) >= 0


def on_edge(p, start, end):
    # collinearity check
    v1 = p - start
    v2 = end - start
    cross = v1[0]*v2[1] - v2[0]*v1[1]

    # check if in line segment
    v1_len_squared = v1 @ v1
    v2_len_squared = v2 @ v2
    return (abs(cross) < 1e-8) and (-1e-8 < v1_len_squared < v2_len_squared + 1e-8)


def on_square_edge(p, s):
    s = np.append(s, [s[0]], axis=0)
    for i in range(s.shape[0] - 1):
        if on_edge(p, s[i], s[i+1]):
            return True
    return False


def remove_in_area_points(points, s):
    inside = np.array([in_area(p, s) and not on_square_edge(p, s) for p in points])
    return points[~inside]


def fill_in_square(s, step_size):
    # travel from edge 1 to edge 3, using known length and direction of edge 4
    start_points = make_line(s[0], s[1], step_size)
    in_area_points = np.array([]).reshape(0, 2)
    for point in start_points:
        in_area_points = np.append(in_area_points, make_line(point, point + (s[3] - s[0]), step_size), axis=0)
    return in_area_points


def fill_in_squares(s1, s2, step_size):
    s1_inner_points = fill_in_square(s1, step_size)
    s2_inner_points = fill_in_square(s2, step_size)
    s2_inner_points = remove_in_area_points(s2_inner_points, s1)
    total_inner_points = np.append(s1_inner_points, s2_inner_points, axis=0)
    return total_inner_points


def fetch_neighbors(p, mesh_points, n):
    tree = KDTree(mesh_points)
    neighbor_distances, neighbor_indices = tree.query(p, k=n)
    mask = np.where(neighbor_distances > 1e-8)[0]
    return np.array([mesh_points[i] for i in neighbor_indices[mask]])