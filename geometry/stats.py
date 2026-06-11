import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d
import shapely
from shapely.plotting import plot_line, plot_polygon
from shapely.geometry import LineString, Polygon
import numpy as np

class Stats:
    def __init__(self, points, mesh: Polygon, buffer=0.01, delaunay_out=None):
        self.buffer = buffer

        self.points = points
        self.mesh = mesh
        self.edges = self.make_delaunay(delaunay_out)


    def make_delaunay(self, out=None):
        tri = Delaunay(self.points)
        triangles = tri.simplices
        edge_indices = np.concatenate([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
        ], axis=0)

        edges = np.zeros((0,2,2))
        for edge_indices in edge_indices:
            edge = np.array([self.points[i] for i in edge_indices])
            line = LineString(edge)
            plot_line(line)
            intersects = line.crosses(self.mesh.boundary)

            if not intersects:
                edges = np.append(edges, np.array([edge]), axis=0)
        
        if out:
            fig, ax = plt.subplots(1,1)
            ax.set_aspect(1)
            delaunay_plot_2d(tri)
            plt.savefig(out)


        plot_polygon(self.mesh)
        plt.show()
        return edges
    

    def make_dists_pdf(self, ax):
        distances = []
        for edge in self.edges:
            distances.append(np.linalg.norm(edge[1] - edge[0]))

        bin_amt = 50 
        x_span = max(distances) * 1.05
        total_samples = 0
        bin_size = x_span / bin_amt
        bins = np.zeros(bin_amt)
        for dist in distances:
            for i in range(bin_amt):
                if i*bin_size <= dist < (i+1)*bin_size:
                    bins[i] += 1
                    total_samples += 1

        # plot pdf
        for i, bin in enumerate(bins):
            ax.bar(i*bin_size, bin / total_samples, width=bin_size, edgecolor="black")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_title("Single Point PDF")
        return ax