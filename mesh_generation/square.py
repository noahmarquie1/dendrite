import numpy as np
from shapely import Polygon, Point, LineString
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt


class Square:
    def __init__(self, width, height, step_size=0.1):

        self.corners = np.array([
            [-width / 2, -height / 2], [width / 2, -height / 2], 
            [width / 2, height / 2], [-width / 2, height / 2]
        ])
        self.step_size = step_size

        n_step_x = round(width / step_size)
        n_step_y = round(height / step_size)

        x_start = np.array([-width / 2, -height / 2])
        x_end = np.array([width / 2, -height / 2])

        y_start = np.array([width / 2, -height / 2])
        y_end = np.array([width / 2, height / 2])

        self.x_pts = np.linspace(x_start, x_end, n_step_x)
        self.y_pts = np.linspace(y_start, y_end, n_step_y)

        self.make_grid()
        self.make_mesh()


    def make_grid(self):
        base = None
        for p1 in self.x_pts:
            for p2 in self.y_pts:
                dist = np.linalg.norm(p1 - p2)
                if dist < 1e-7:
                    base = p1
                    break;

        if base is None:
            raise ValueError("Axes not aligned.")

        x = self.x_pts[np.newaxis, :, :] # x points form columns
        y = self.y_pts[:, np.newaxis, :] # y points form rows
        self.grid = x + y - base

        self.rows = [self.grid[i, :, :] for i in range(len(self.y_pts))]
        self.cols = [self.grid[:, i, :] for i in range(len(self.x_pts))]
        self.points = np.vstack([row for row in self.rows])


    def make_mesh(self):
        self.mesh = Polygon(self.corners)


    def transform(self, offset, theta):
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])

        self.points = self.points @ rotation_matrix.T
        self.points[:, 0] += offset[0]
        self.points[:, 1] += offset[1]

        self.corners = self.corners @ rotation_matrix.T
        self.corners[:, 0] += offset[0]
        self.corners[:, 1] += offset[1]

        self.make_grid()
        self.make_mesh()

    
    def add_edge_point(self, point: Point):
        line = None
        self.points = np.zeros((0,2))

        b = LineString([self.corners[0], self.corners[1]])
        r = LineString([self.corners[1], self.corners[2]])
        t = LineString([self.corners[3], self.corners[2]])
        l = LineString([self.corners[0], self.corners[3]])
        edges = [b, r, t, l]
        axes = { b: 'x_pts', t: 'x_pts', l: 'y_pts', r: 'y_pts' }
        tolerance = 1e-7

        edge_found = False
        for edge in edges:
            if edge.distance(point) < tolerance:
                edge_found = True
                start = np.array(edge.coords[0])
                end = np.array(edge.coords[1])
                point = edge.interpolate(edge.project(point))
                point_np = np.array(point.coords).reshape(2,)

                n_step = round(np.linalg.norm(start - point_np) / self.step_size)
                line = np.vstack([np.linspace(start, point_np, n_step), np.linspace(point_np, end, n_step)])
                setattr(self, axes[edge], line)
                break
        
        if not edge_found:
            raise ValueError("Point entered is not on an edge of the square")

        self.make_grid()
        self.make_mesh()




if __name__ == "__main__":
    square = Square(1, 2, step_size=0.2)
    square.add_edge_point(Point([0.5, 0.1]))

    for row in square.rows:
        plt.scatter(row[:, 0], row[:, 1])

    plot_polygon(square.mesh)
    plt.show()