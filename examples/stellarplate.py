import polars as pl
import numpy as np
from geometry.hex_geometry import Hexagon
import matplotlib.pyplot as plt

data = pl.read_csv("data/stellarplate.csv")
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(8)

ax.set_aspect(1)

for row in data.iter_rows():
    radius, trans_x, trans_y, theta = row
    hex = Hexagon(radius, 0.0005)
    hex.transform((trans_x, trans_y), theta)
    hex.visualize(ax)

plt.show()
