import numpy as np
from shapely.geometry import Polygon, Point
from geometry.base_geometry import Edge, Shape, transform_points


# Rect Class
class Rect(Shape):
    def __init__(self, width, height, step_size=0.1):

        self.corners = np.array([
            [-width / 2, -height / 2], [width / 2, -height / 2], 
            [width / 2, height / 2], [-width / 2, height / 2]
        ])
        self.step_size = step_size

        n_step_x = round(width / step_size)
        n_step_y = round(height / step_size)

        bl = np.array([-width / 2, -height / 2])
        br = np.array([width / 2, -height / 2])
        tr = np.array([width / 2, height / 2])

        self.x_pts = np.linspace(bl, br, n_step_x)
        self.y_pts = np.linspace(br, tr, n_step_y)
        self.edge_points = np.zeros((0,2))
        self.update_geometry()


    def update_geometry(self):
        self.update_edges()
        self.make_grid()
        self.make_mesh()


    def update_edges(self):
        self.b = Edge(self.corners[0], self.corners[1])
        self.r = Edge(self.corners[1], self.corners[2])
        self.t = Edge(self.corners[3], self.corners[2])
        self.l = Edge(self.corners[0], self.corners[3])
        self.edges = [self.b, self.r, self.t, self.l]

        self.opposite_edges = { self.b: self.t, self.t: self.b, self.l: self.r, self.r: self.l }
        self.axes = { self.b: 'x_pts', self.t: 'x_pts', self.l: 'y_pts', self.r: 'y_pts' }


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
        self.edge_points = np.vstack([
            self.grid[0, :, :],
            self.grid[-1, :, :],
            self.grid[:, 0, :],
            self.grid[:, -1, :],
        ])
        self.edge_points = np.unique(self.edge_points, axis=0)
        self.points = np.vstack([self.grid[i, :, :] for i in range(len(self.y_pts))])


    def make_mesh(self):
        self.mesh = Polygon(self.corners)


    def transform(self, offset, theta):
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])

        self.points = transform_points(self.points, offset, rotation_matrix)
        self.x_pts = transform_points(self.x_pts, offset, rotation_matrix)
        self.y_pts = transform_points(self.y_pts, offset, rotation_matrix)
        self.corners = transform_points(self.corners, offset, rotation_matrix)
        self.update_geometry()

    
    def add_edge_point(self, point: Point):
        line_points = np.zeros((0,2))
        self.points = np.zeros((0,2))
        tolerance = 1e-7

        edge_found = False
        for edge in self.edges:
            if edge.total_linestring.distance(point) < tolerance:
                edge_found = True
                edge.add_point(point)

                opposite_edge = self.opposite_edges[edge]
                opposite_point = opposite_edge.total_linestring.interpolate(opposite_edge.total_linestring.project(point))
                opposite_edge.add_point(opposite_point)
                line_points = np.append(line_points, np.array([edge.linestrings[0].coords[0]]), axis=0)

                for linestring in edge.linestrings:
                    start = np.array(linestring.coords)[0]
                    end = np.array(linestring.coords)[1]
                    n_step = max(3, round(np.linalg.norm(end - start) / self.step_size))
                    line_seg = np.linspace(start, end, n_step)[1:] 
                    line_points = np.append(line_points, line_seg, axis=0)

                setattr(self, self.axes[edge], line_points)
                break;
        
        if not edge_found:
            raise ValueError("Point entered is not on an edge of the square")

        self.make_grid()
        self.make_mesh()