from hmac import digest_size

from geometry import *
from stats import *
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon

s1 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]) # simple square between 0 and 1
s3 = np.array([[1, 0], [0, 1], [1, 2], [2, 1]])
approx_step = 0.1
mesh_points = np.array([]).reshape(0, 2)

mesh_points = np.append(mesh_points, s1, axis=0)
mesh_points = np.append(mesh_points, s3, axis=0)
mesh_points = np.append(mesh_points, make_square_edges(s1, approx_step), axis=0)
mesh_points = np.append(mesh_points, make_square_edges(s3, approx_step), axis=0)

mesh_points = remove_in_area_points(mesh_points, s1)
mesh_points = remove_in_area_points(mesh_points, s3)
mesh_points = np.append(mesh_points, fill_in_squares(s1, s3, approx_step), axis=0)

sample_idx = 200
sample_point = mesh_points[sample_idx]

_, ax = plt.subplots(2, 1, figsize=(6, 8))
plot_single_point_pdf(sample_point, mesh_points, approx_step, ax=ax[0])
plot_mesh_pdf(mesh_points, approx_step, ax=ax[1])
ax[0].set_title("Single Point PDF")
ax[1].set_title("Mesh PDF")
plt.tight_layout()
plt.show()

# plot original shape and mesh side-by-side
_, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_polygon(Polygon(s1), ax=ax[0])
plot_polygon(Polygon(s3), ax=ax[0])

single_point_neighbors = fetch_neighbors(sample_point, mesh_points, 8)
ax[1].scatter(mesh_points[:, 0], mesh_points[:, 1], color='lightblue', marker='.')
ax[1].scatter(single_point_neighbors[:, 0], single_point_neighbors[:, 1], color='orange')
ax[1].scatter(sample_point[0], sample_point[1], color='red')
plt.tight_layout()
plt.show()