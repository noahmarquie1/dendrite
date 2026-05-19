from mesh_generation.snowflake_mesh import generate_full_snowflake
from particle_sim.solver import PointCloudSolver
import numpy as np

N_BODIES = 16
FORCE_MULTIPLIER = 10
DPI = 75
DRAG_COEF = 2

snowflake_mesh = generate_full_snowflake(1, 2, 3, 0.1)

width = snowflake_mesh.polygon.bounds[2] - snowflake_mesh.polygon.bounds[0]
height = snowflake_mesh.polygon.bounds[3] - snowflake_mesh.polygon.bounds[1]

T = 6
step = T * 1e-2

solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    force_multiplier=FORCE_MULTIPLIER,
    drag_coeff=DRAG_COEF,
    plots=['pdf-anim', 'max-vel', 'pdf-final'],
    polygon=snowflake_mesh.polygon,
)

solver.solve(
    max_step=step,
    steps=600, 
    out="anim.gif"
    #state0=state0 # optional, specifies hexagonal initial state
)
solver.animate()

