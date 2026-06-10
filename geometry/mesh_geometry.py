import numpy as np
import shapely
from shapely import Polygon, Point
from shapely.plotting import plot_polygon
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from simulation.solver import PointCloudSolver
import jax.numpy as jnp
from geometry.rect_geometry import Rect


# Static Region Class
#  - starts off with points from a Rect
#  - points can be refreshed when a new square is added (often updating grid layouts)
#  - keeps track of intersecting dynamic regions as to avoid duplicate points
class StaticRegion:
    def __init__(self, rect: Rect):
        self.points = rect.points
        self.boundary_points = rect.edge_points
        self.mesh = shapely.convex_hull(Polygon(self.points))


    def visualize(self, ax):
        ax.scatter(self.points[:, 0], self.points[:, 1], alpha=0.3, s=8, c='blue')


# Dynamic Region Classes
class DynamicRegionSolver(PointCloudSolver):
    def __init__(self, dpi=100, width=6, height=6, n_bodies=1, polygon=None, fps=15, region=None):
        super().__init__(dpi=dpi, width=width, height=height, n_bodies=n_bodies, polygon=polygon, fps=fps)
        self.region = region


    def calculate_derivatives(self, state):
        state = state.reshape(-1, 2)
        n_bodies = int(state.shape[0] / 2)
        pos_i = state[:n_bodies]
        pos_i = jnp.vstack([pos_i, self.region.boundary_points])
        vel_i = state[n_bodies:]

        delta = pos_i[:, None, :] - pos_i[None, :, :]
        p_forces = self.point_vmap(delta)
        p_forces = jnp.sum(p_forces, axis=1) / 2
        p_forces = p_forces[:n_bodies]

        w_forces = self.wall_vmap(pos_i)[:n_bodies]
        drag = self.drag_vmap(vel_i)
        total_force = p_forces + drag + w_forces

        return jnp.vstack([vel_i, total_force]).flatten()



class DynamicRegion:
    def __init__(self, boundary_points, connecting_points, n_bodies):
        self.boundary_points = boundary_points
        self.connecting_points = connecting_points
        self.boundary_points = np.append(self.boundary_points, self.connecting_points, axis=0)

        self.mesh = shapely.convex_hull(Polygon(boundary_points))
        self.n_bodies = n_bodies
        self.solver: DynamicRegionSolver = DynamicRegionSolver(polygon=self.mesh, n_bodies=self.n_bodies, region=self)
        self.filled_points = None

    def visualize(self):
        plt.close()
        plot_polygon(self.mesh)
        plt.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], c='red')
        plt.scatter(self.filled_points[:, 0], self.filled_points[:, 1], c='blue')

    
    def set_n_bodies(self, n_bodies):
        self.n_bodies = n_bodies
        self.solver.n_bodies = n_bodies


    def fill(self, verbose=0):
        if self.boundary_points.shape[0] > 6:
            sol = self.solver.solve(steps=int(3e3))
            if verbose:
                self.solver.animate("anim.mp4")
            self.filled_points = sol[-1][:self.n_bodies]
        else:
            self.filled_points = np.zeros((0,2))


# Mesh Class
class Mesh:
    def __init__(self, rect):
        self.rects: list[Rect] = [rect] 
        self.mesh = rect.mesh
        self.boundary_points = rect.edge_points
        self.static_regions: dict[Rect, StaticRegion] = { rect: StaticRegion(rect=rect) }
        self.intersections: dict[Rect, list[DynamicRegion]] = { rect: [] }
        self.dynamic_regions: list[DynamicRegion] = []


    def padded_intersection(self, r1: Rect, r2: Rect, pad=0):
        intersection = r1.mesh.intersection(r2.mesh)
        padding_width = pad * min(r1.step_size, r2.step_size)
        return intersection.buffer(padding_width)

    
    def update_static_regions(self, r1: Rect, r2: Rect):
        intersection = r1.mesh.intersection(r2.mesh)
        r1_boundary_points = self.remove_intersecting_points(self.static_regions[r1].boundary_points, intersection)
        r2_boundary_points = self.remove_intersecting_points(self.static_regions[r2].boundary_points, intersection)

        padded_intersection = self.padded_intersection(r1, r2, pad=1.5)
        r1_points = self.remove_intersecting_points(self.static_regions[r1].points, padded_intersection)
        r2_points = self.remove_intersecting_points(self.static_regions[r2].points, padded_intersection)

        self.static_regions[r1].points = r1_points
        self.static_regions[r2].points = r2_points

        self.static_regions[r1].boundary_points = r1_boundary_points
        self.static_regions[r2].boundary_points = r2_boundary_points

        self.boundary_points = np.vstack([region.boundary_points for region in self.static_regions.values()])

        
    def add_intersection(self, rects, dynamic_region):
        for rect in rects:
            if not rect in self.intersections.keys():
                self.intersections[rect] = []

            self.intersections[rect].append(dynamic_region)
                    
    
    def pt_in_array(self, pt, arr):
        return any(np.allclose(pt, arr_p, rtol=1e-4, atol=1e-4) for arr_p in arr)


    def remove_intersecting_points(self, points, shape: Polygon):
        end_points = []
        for point in points:
            if not shape.distance(Point(point)) < 1e-7:
                end_points.append(point)

        return np.array(end_points)


    def create_dynamic_region(self, r1: Rect, r2: Rect, connecting_points) -> DynamicRegion:
        step_size = min(r1.step_size, r2.step_size)
        intersection = self.padded_intersection(r1, r2, 1.5)
        s1_overlapping = [point for point in r1.points if intersection.distance(Point(point)) < 0.5*step_size]
        s2_overlapping = [point for point in r2.points if intersection.distance(Point(point)) < 0.5*step_size]
        overlapping = np.vstack([s1_overlapping, s2_overlapping])

        non_overlapping = np.vstack([
            self.static_regions[r1].points,
            self.static_regions[r2].points
        ])

        # Initial Boundary Creation
        tree_overlap = KDTree(overlapping)
        dists_to_overlap, _ = tree_overlap.query(non_overlapping)
        boundary_mask = (dists_to_overlap <= step_size * 1.5)
        boundary_points = non_overlapping[boundary_mask]
        self.boundary_points = np.append(self.boundary_points, connecting_points, axis=0)

        # Boundary Shrinking
        square_bounds = np.vstack([
            self.static_regions[r1].boundary_points,
            self.static_regions[r2].boundary_points
        ])
        square_bounds = np.array([point for point in square_bounds if intersection.contains(Point(point))])
        boundary_points = np.append(boundary_points, square_bounds, axis=0)

        connecting_points = np.append(connecting_points, square_bounds, axis=0)

        n_bodies = int(overlapping.shape[0] / 2)
        dynamic_region = DynamicRegion(boundary_points=boundary_points, connecting_points=connecting_points, n_bodies=n_bodies)
        self.dynamic_regions.append(dynamic_region)
        self.add_intersection([r1, r2], dynamic_region)

        return dynamic_region
        

    def add_rect(self, rect_n: Rect, verbose=0): # Main function used outside class
        self.static_regions[rect_n] = StaticRegion(rect_n)

        for rect in self.rects:
            edges1 = rect.mesh.exterior
            edges2 = rect_n.mesh.exterior

            intersection = edges2.intersection(edges1)

            connecting_points = shapely.get_coordinates(intersection)

            if connecting_points.shape[0] > 0:
                for point in connecting_points:
                    rect.add_edge_point(Point(point))
                    rect_n.add_edge_point(Point(point))

                if verbose:
                    print("Updating static regions")
                self.update_static_regions(rect, rect_n)
                if verbose:
                    print("Creating dynamic region")
                self.create_dynamic_region(rect, rect_n, connecting_points)
                
        self.rects.append(rect_n)


    def dynamic_fill(self, verbose=0):
        for i, region in enumerate(self.dynamic_regions):
            print(f"Filling region {i}/{len(self.dynamic_regions)}")
            region.fill(verbose=verbose)


    def animate_region(self, dynamic_region: DynamicRegion, debug=False):
        plt.close('all')
        plt.style.use('seaborn-v0_8-paper')
        anim = dynamic_region.solver.anim
        anim.out = "anim.gif"
        anim.configure_plot()

        for static_region in self.static_regions.values():
            anim.add_static_points(static_region.points)

        if debug:
            #plot_polygon(dynamic_region.mesh)
            plt.scatter(dynamic_region.boundary_points[:, 0], dynamic_region.boundary_points[:, 1], c="red")
            anim.primary_ax.set_xlim(dynamic_region.mesh.bounds[0], dynamic_region.mesh.bounds[2])
            anim.primary_ax.set_ylim(dynamic_region.mesh.bounds[1], dynamic_region.mesh.bounds[3])
        
        anim.animate(dynamic_region.solver.solution)

