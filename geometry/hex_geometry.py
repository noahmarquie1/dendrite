from tri_geometry import Triangle
import numpy as np
import matplotlib.pyplot as plt

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


class Hexagon:
    def __init__(self, radius, step_size=0.01):
        tri_1 = Triangle(unit_hex[0], step_size)
        tri_2 = Triangle(unit_hex[1], step_size)
        tri_3 = Triangle(unit_hex[2], step_size)

        tri_4 = Triangle(unit_hex[3], step_size)
        tri_5 = Triangle(unit_hex[4], step_size)
        tri_6 = Triangle(unit_hex[5], step_size)

        self.triangles = [tri_1, tri_2, tri_3, tri_4, tri_5, tri_6]


    def visualize(self):
        for tri in self.triangles:
            tri.visualize()


hex = Hexagon(1, step_size = 0.1)
hex.visualize()
plt.show()
