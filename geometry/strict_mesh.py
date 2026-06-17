from geometry.mesh_geometry import Mesh, StaticRegion, DynamicRegion
from geometry.rect_geometry import Rect
from geometry.base_geometry import Shape
from shapely.plotting import plot_polygon
import shapely
from shapely import Point
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


def new_unique_points(new, orig):
    tree = KDTree(orig)
    dists, _ = tree.query(new)
    mask = dists > 1e-8
    return new[mask]


class StrictMesh(Mesh):
    def __init__(self, shapes: list[Shape], dynamic=True):
        # Setup first shape
        super().__init__(shapes[0])
        orig_static = self.static_regions[shapes[0]]
        mask = shapely.dwithin(orig_static.mesh.boundary, shapely.points(orig_static.points), distance=1e-8)
        orig_static.points = orig_static.points[~mask]

        self.shared_boundary_points = np.zeros((0,2))

        for i in range(1, len(shapes)):
            self.add_shape(shapes[i])

        mask = shapely.dwithin(self.mesh.boundary, shapely.points(self.shared_boundary_points), distance=1e-8)
        self.shared_boundary_points = self.shared_boundary_points[mask]

        self.global_inner_points = np.unique(
            np.vstack([region.points for region in self.static_regions.values()]), axis=0
        )
        self.global_boundary_points = np.unique(
            np.vstack([region.boundary_points for region in self.static_regions.values()]),
            axis=0
        )
        self.global_boundary_points = np.vstack([self.global_boundary_points, self.shared_boundary_points])

        # Setup dynamic regions
        self.intersections = self.get_intersections()
        if dynamic:
            self.make_dynamic()


    def add_shape(self, shape_n: Shape, verbose=0):
        self.mesh = shapely.union(self.mesh, shape_n.mesh)
        boundary_points = shape_n.edge_points
        inner_points = new_unique_points(shape_n.points, boundary_points)

        for i in range(0, len(self.shapes)):
            shape = self.shapes[i]

            edges1 = shape.mesh.exterior
            edges2 = shape_n.mesh.exterior
            intersection = edges2.intersection(edges1)
            connecting_points = shapely.get_coordinates(intersection)

            self.shared_boundary_points = np.vstack([self.shared_boundary_points, new_unique_points(new=connecting_points, orig=self.shared_boundary_points)])

            if connecting_points.shape[0] > 0:
                for point in connecting_points:
                    shape.add_edge_point(point)
                    shape_n.add_edge_point(point)

                # Edit static region for new rect
                min_point_dist = 3e-5
                bounds_mask = (
                    shapely.dwithin(shape.mesh, shapely.points(boundary_points), distance=1e-8)
                    | shapely.dwithin(shapely.MultiPoint(connecting_points), shapely.points(boundary_points), distance=min_point_dist)
                )
                inner_mask = shapely.dwithin(shape.mesh, shapely.points(inner_points), distance=1e-8)
                boundary_points = boundary_points[~bounds_mask]
                inner_points = inner_points[~inner_mask]

                # Edit static region for old rect
                region = self.static_regions[shape]
                bounds_mask = (
                    shapely.dwithin(shape_n.mesh, shapely.points(region.boundary_points), distance=1e-8)
                    | shapely.dwithin(shapely.MultiPoint(connecting_points), shapely.points(region.boundary_points), distance=min_point_dist)
                )
                inner_mask = shapely.dwithin(shape_n.mesh, shapely.points(region.points), distance=1e-8)
                region.boundary_points = region.boundary_points[~bounds_mask]
                region.points = region.points[~inner_mask]

        static_region = StaticRegion(shape_n, inner_points, boundary_points)
        self.static_regions[shape_n] = static_region
        self.shapes.append(shape_n)


    def get_intersections(self):
        intersections = []
        for i in range(0, len(self.shapes)):
            for j in range(i+1, len(self.shapes)):
                intersection = shapely.intersection(self.shapes[i].mesh, self.shapes[j].mesh)
                if not intersection.is_empty:
                    intersections.append(intersection)

        unions = []
        isolated = []
        while len(intersections) > 0:
            intersect_indices = []
            current_union = None
            int1 = intersections[0]

            for i in range(1, len(intersections)):
                int2 = intersections[i]
                if not shapely.intersection(int1, int2).is_empty:
                    intersect_indices.append(i)
                    if current_union is None:
                        current_union = shapely.union(int1, int2)
                    else:
                        current_union = shapely.union(current_union, int2)

            if len(intersect_indices) > 0:
                intersect_indices.reverse() # pop in reverse order
                for idx in intersect_indices:
                    intersections.pop(idx)

            if not current_union is None:
                unions.append(current_union)
            else:
                isolated.append(intersections[0])

            intersections.pop(0)
        
        return unions + isolated
    

    def make_dynamic(self):
        for intersection in self.intersections:
            padding_width = 2 * self.shapes[0].step_size
            padded_intersection = intersection.buffer(padding_width)

            all_points = np.vstack([
                self.global_boundary_points,
                self.global_inner_points
            ])
            mask = shapely.dwithin(padded_intersection, shapely.points(all_points), distance=1e-8)
            dynamic_bound_points = all_points[mask]

            n_bodies = int(2 * shapely.area(intersection) / (np.sqrt(3) * (self.shapes[0].step_size ** 2)))
            dynamic_region = DynamicRegion(dynamic_bound_points, np.zeros((0,2)), n_bodies, padded_intersection)
            self.dynamic_regions.append(dynamic_region)

            dynamic_region.fill()
            
            # Mask out misbehaving points if necessary
            mask = shapely.dwithin(intersection, shapely.points(dynamic_region.filled_points), distance=3e-5)
            dynamic_region.filled_points = dynamic_region.filled_points[mask]

    
    def visualize(self, ax, verbose=0):
        
        ax.scatter(self.global_boundary_points[:, 0], self.global_boundary_points[:, 1], alpha=0.3)
        ax.scatter(self.global_inner_points[:, 0], self.global_inner_points[:, 1], alpha=0.3)

        for region in self.dynamic_regions:
            ax.scatter(region.filled_points[:, 0], region.filled_points[:, 1], alpha=0.3)
            if verbose:
                plot_polygon(region.mesh)
                ax.scatter(region.boundary_points[:, 0], region.boundary_points[:, 1], alpha=1, c='red')


