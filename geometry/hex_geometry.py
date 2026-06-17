from geometry.tri_geometry import Triangle
from geometry.base_geometry import Shape
import numpy as np
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely import Point, Polygon
import shapely

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


class Hexagon(Shape):
    def __init__(self, radius, step_size=0.01):
        self.step_size = step_size
        self.tri_1 = Triangle(unit_hex[0] * radius, self.step_size)
        self.tri_2 = Triangle(unit_hex[1] * radius, self.step_size)
        self.tri_3 = Triangle(unit_hex[2] * radius, self.step_size)

        self.tri_4 = Triangle(unit_hex[3] * radius, self.step_size)
        self.tri_5 = Triangle(unit_hex[4] * radius, self.step_size)
        self.tri_6 = Triangle(unit_hex[5] * radius, self.step_size)

        self.triangles: list[Triangle] = [self.tri_1, self.tri_2, self.tri_3, self.tri_4, self.tri_5, self.tri_6]
        self.load_hex()


    def load_hex(self):
        self.points = np.vstack([
            self.tri_1.inner_points,
            self.tri_2.inner_points,
            self.tri_3.inner_points,

            self.tri_4.inner_points,
            self.tri_5.inner_points,
            self.tri_6.inner_points,
            
            self.tri_1.edge2.points, self.tri_1.edge2.end, 
            self.tri_2.edge1.points,
            self.tri_3.edge1.points,
            self.tri_4.edge3.points,
            self.tri_5.edge3.points,
            self.tri_6.edge2.points,
        ])

        self.edge_points = np.vstack([
            self.tri_1.edge1.points, self.tri_1.edge1.start,
            self.tri_2.edge3.points, self.tri_2.edge3.end,
            self.tri_3.edge3.points, self.tri_3.edge3.end,
            self.tri_4.edge2.points, self.tri_4.edge2.end,
            self.tri_5.edge2.points, self.tri_5.edge2.end,
            self.tri_6.edge1.points, self.tri_6.edge1.start,
        ])

        self.mesh = shapely.convex_hull(Polygon(self.edge_points))


    def transform(self, offset, theta):
        for tri in self.triangles:
            tri.transform(offset, theta)
        self.load_hex()


    def add_edge_point(self, point):
        tolerance = 1e-5

        for i, tri in enumerate(self.triangles):
            if tri.mesh.distance(Point(point)) < tolerance:
                tri.add_edge_point(point)
                break;
    
        self.load_hex()



    def visualize(self, ax):
        ax.scatter(self.points[:, 0], self.points[:, 1], alpha=0.5)
        ax.scatter(self.edge_points[:, 0], self.edge_points[:, 1], alpha=0.5, c="red")


if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    ax.set_aspect(1)
    hex = Hexagon(1, step_size = 0.1)

    new_point = np.array([[0, np.sqrt(3/4)]])
    hex.add_edge_point(new_point)
    hex.visualize(ax)
    plt.show()
