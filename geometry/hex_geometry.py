from tri_geometry import Triangle
import numpy as np
import matplotlib.pyplot as plt

unit_hex = np.array([
    # Top half of hexagon
    [[-1, 0], [-0.5, np.sqrt(3/4)], [0, 0]],            
    [[0, 0], [-0.5, np.sqrt(3/4)], [0.5, np.sqrt(3/4)]],
    [[0, 0], [0.5, np.sqrt(3/4)], [1, 0]],

    # bottom half of hexagon
    [[0.5, -np.sqrt(3/4)], [0, 0], [1, 0]],
    [[-0.5, -np.sqrt(3/4)], [0, 0], [0.5, -np.sqrt(3/4)]],
    [[-0.5, -np.sqrt(3/4)], [-1, 0], [0, 0]],
])


class Hexagon:
    def __init__(self, radius, step_size=0.01):
        tri_1 = Triangle(unit_hex[0] * radius, step_size)
        tri_2 = Triangle(unit_hex[1] * radius, step_size)
        tri_3 = Triangle(unit_hex[2] * radius, step_size)

        tri_4 = Triangle(unit_hex[3] * radius, step_size)
        tri_5 = Triangle(unit_hex[4] * radius, step_size)
        tri_6 = Triangle(unit_hex[5] * radius, step_size)

        self.triangles = [tri_1, tri_2, tri_3, tri_4, tri_5, tri_6]

        # Put together boundary and inner points from each triangle
        self.inner_points = np.vstack([
            tri_1.inner_points,
            tri_2.inner_points,
            tri_3.inner_points,

            tri_4.inner_points,
            tri_5.inner_points,
            tri_6.inner_points,
            
            tri_1.edge2.points, tri_1.edge2.end, 
            tri_2.edge1.points,
            tri_3.edge1.points,
            tri_4.edge3.points,
            tri_5.edge3.points,
            tri_6.edge2.points,
        ])

        self.boundary_points = np.vstack([
            tri_1.edge1.points, tri_1.edge1.start,
            tri_2.edge3.points, tri_2.edge3.end,
            tri_3.edge3.points, tri_3.edge3.end,
            tri_4.edge2.points, tri_4.edge2.end,
            tri_5.edge2.points, tri_5.edge2.end,
            tri_6.edge1.points, tri_6.edge1.start,
        ])


    def visualize(self):
        plt.scatter(self.inner_points[:, 0], self.inner_points[:, 1], alpha=0.5)
        plt.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], alpha=0.5, c="red")


if __name__ == "__main__":
    hex = Hexagon(1, step_size = 0.1)
    hex.visualize()
    plt.show()
