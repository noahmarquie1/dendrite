import shapely
import numpy as np
from scipy.ndimage import distance_transform_edt
from jax.scipy.ndimage import map_coordinates

def generate_sdf(geometry, res=256):
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    xv, yv = np.meshgrid(x, y)
    
    points = shapely.points(xv, yv)
    mask = shapely.contains(geometry, points)
    dx = 1.0 / (res - 1)

    dist_outside = distance_transform_edt(~mask, sampling=dx)
    dist_inside = distance_transform_edt(mask, sampling=dx)
    sdf_grid = dist_outside - dist_inside
    grad_y, grad_x = np.gradient(sdf_grid, dx)
    
    return sdf_grid, grad_x, grad_y


def sample_sdf(grid, this):
    res = grid.shape[0]
    iy = this[1] * (res - 1)
    ix = this[0] * (res - 1)

    return map_coordinates(input=grid, coordinates=(iy, ix), order=1, mode='nearest')