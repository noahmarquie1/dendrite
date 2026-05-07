from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np

def get_edges(polygon):
    edges = np.zeros((0, 2, 2))
    for i, coord in enumerate(polygon.exterior.coords):
        if (i == len(polygon.exterior.coords) - 1):
            edge =  np.array([[coord, polygon.exterior.coords[0]]])
        else:
            edge =  np.array([[coord, polygon.exterior.coords[i+1]]])
        edges = np.append(edges, edge, axis=0)
    return edges

def find_closest_point_on_edge(point, edge_start, edge_end):
    p = Point(point)
    l = LineString([edge_start, edge_end])
    closest = nearest_points(l, p)[0]
    return np.array(closest.coords[0])


def find_nearest_edge(point, edges):
    best_distance = float('inf')
    best_edge = None
    for edge in edges:
        closest_points = find_closest_point_on_edge(point, edge[0], edge[1])
        dist = np.linalg.norm(point - closest_points)
        if dist < best_distance:
            best_distance = dist
            best_edge = edge
    return best_edge, best_distance
