from mesh_generation.snowflake_mesh import generate_full_snowflake
from particle_sim.solver import PointCloudSolver
import numpy as np

N_BODIES_IDEAL = 100
FORCE_MULTIPLIER = 10
DPI = 75
DRAG_COEF = 2

snowflake_mesh = generate_full_snowflake(1, 2, 3, 0.1)

width = snowflake_mesh.polygon.bounds[2] - snowflake_mesh.polygon.bounds[0]
height = snowflake_mesh.polygon.bounds[3] - snowflake_mesh.polygon.bounds[1]

snowflake_mesh.step_size = max(width, height) / 4
initial_points = np.zeros((0, 2))
while initial_points.shape[0] < N_BODIES_IDEAL:
    initial_points = snowflake_mesh.hex_fill()
    snowflake_mesh.step_size *= 0.95

print("Initial points intended: ", N_BODIES_IDEAL)
print("Initial points generated:", initial_points.shape[0])
n_bodies_actual = initial_points.shape[0]

vel = np.zeros((n_bodies_actual, 2))
state0 = np.concatenate((initial_points, vel))

solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=n_bodies_actual,
    force_multiplier=FORCE_MULTIPLIER,
    width=width,
    height=height,
    drag_coefficient=DRAG_COEF,
    pdfs=True,
    pdf_interval=50,
    polygon=snowflake_mesh.polygon,
)

solver.solve(
    h=0.01, 
    steps=600, 
    #state0=state0 # optional, specifies hexagonal initial state
)
solver.animate()

