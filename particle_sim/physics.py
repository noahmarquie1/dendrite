import numpy as np
from random import choice
from numpy.char import center
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from particle_sim.geometry import get_edges, find_closest_point_on_edge, find_nearest_edge

class PhysicsHandler:
    def __init__(self, n_bodies, force_multiplier=100, drag_coefficient=0.01, width=6, height=4, polygon: Polygon=None):
        self.n_bodies = n_bodies
        self.force_multiplier = force_multiplier
        self.drag_coefficient = drag_coefficient
        self.width = width
        self.height = height

        if polygon is None:
            self.polygon = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        else:
            self.polygon = polygon

        self.ideal_dist = 0.75 * np.sqrt(self.polygon.area / self.n_bodies)

    def unit_vector(self, start, end):
        return (end - start) / np.linalg.norm(end - start)


    def calculate_repulsive_force(self, this, other):
        delta = this - other # points away from other
        dist = np.linalg.norm(this - other)
        if dist < 1e-4:
            force = np.array([choice([0.05, -0.05]), choice([0.05, -0.05])])
        else:
            force = self.force_multiplier * delta / (dist ** 3)
        return force
    

    def calculate_morse_law_force(self, this, other):
        dir = self.unit_vector(start=this, end=other)
        dist = np.linalg.norm(this - other)
        s = 1 # sensitivity
        morse_factor = (1-np.exp(-s * (dist - self.ideal_dist))) * np.exp(-s * (dist - self.ideal_dist))
        magnitude = 2 * s * self.force_multiplier * morse_factor
        return magnitude * dir # if >ideal_dist, attract toward other, if <ideal_dist, repel away from other


    def calculate_wall_force(self, this):
        p = Point(this)
        if self.polygon.contains(p):
            if self.polygon.exterior.distance(p) < 0.2:
                closest = self.polygon.exterior.interpolate(self.polygon.exterior.project(p))
                closest = closest.coords[0]
                return self.calculate_morse_law_force(this=this, other=closest)
            return 0
        else:
            center = np.array([self.polygon.centroid.x, self.polygon.centroid.y])
            dir = (center - this) / np.linalg.norm(center - this)
            return 10 * self.force_multiplier * dir


    def calculate_linear_force(self, this, other):
        delta = other - this # points toward other
        dist = np.linalg.norm(delta)
        if dist < 1e-5:
            return (np.random.rand(2) - 0.5) * 0.1

        dir = delta / dist
        ideal_dist = np.sqrt(self.width * self.height / self.n_bodies)

        magnitude = (dist - ideal_dist) / ideal_dist
        magnitude = np.clip(magnitude, -10, 0)
        return self.force_multiplier * magnitude * dir


    def calculate_drag_force(self, vel):
        speed = np.linalg.norm(vel)
        return -self.drag_coefficient * vel * speed