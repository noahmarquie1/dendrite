import numpy as np
from base_geometry import Edge
import matplotlib.pyplot as plt
from shapely import LineString, Point, Polygon
from shapely.plotting import plot_line, plot_points

plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(1,1)
ax.set_aspect(1)

class Triangle:
    def __init__(self, points: np.ndarray, step_size=0.01):
        if points.shape != (3,2):
            print("Please input an array of shape (3,2) to represent a triangle")
        
        self.step_size = step_size

        # Edge objects keep track of permanent points, edge dict m
        self.edge1 = Edge(points[0], points[1], self.step_size)
        self.edge2 = Edge(points[0], points[2], self.step_size)
        self.edge3 = Edge(points[2], points[1], self.step_size)
        self.edges = [self.edge1, self.edge2, self.edge3]
        self.corners = points

        self.inner_points = np.zeros((0,2))
        self.mesh = Polygon(points)
        
        self.create_grid()

    
    def create_grid(self):
        lines_12: list[LineString] = []
        lines_13: list[LineString] = []

        for i in range(len(self.edge1.points)):
            l_12 = LineString([self.edge1.points[i], self.edge2.points[i]])
            l_13 = LineString([self.edge1.points[i], self.edge3.points[i]])
            lines_12.append(l_12)
            lines_13.append(l_13)

        intersections = []
        for i, l1 in enumerate(lines_12):
            for j, l2 in enumerate(lines_13):
                if i != j and type(l1.intersection(l2)) == Point:
                    intersections.append(np.array(l1.intersection(l2).coords))

        if len(intersections) > 0:
            self.inner_points = np.vstack(intersections)
        else:
            self.inner_points = np.zeros((0,2))


    def add_edge_point(self, point: np.ndarray):
        tolerance = 1e-5
        dist_pct = 0
        orig_idx = 0

        print([len(edge.points) for edge in self.edges])

        for i, edge in enumerate(self.edges):
            if edge.total_linestring.distance(Point(point)) < tolerance:
                dist = np.linalg.norm(edge.start - point)
                dist_pct = dist / edge.total_linestring.length
                edge.add_point(Point(point))
                orig_idx = i
                print(edge.points.shape)
                break;
    
        for i, edge in enumerate(self.edges):
            if i != orig_idx:
                new_point = edge.total_linestring.interpolate(dist_pct * edge.total_linestring.length)
                edge.add_point(new_point)
                print(edge.points.shape)

        print([len(edge.points) for edge in self.edges])

        self.create_grid()
                

    def visualize(self):
        for edge in self.edges:
            print("edge found")
            edge_points = np.array(edge.points)
            plt.scatter(edge_points[:, 0], edge_points[:, 1], alpha=0.4)

        plt.scatter(self.inner_points[:, 0], self.inner_points[:, 1], alpha=0.4)
        plt.scatter(self.corners[:, 0], self.corners[:, 1], alpha=0.4)


if __name__ == "__main__":
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3/4)],
    ])

    tri = Triangle(points, step_size=0.1)

    edge_point = np.array([0.45, 0])
    tri.add_edge_point(edge_point)
    tri.visualize()
    plt.show()