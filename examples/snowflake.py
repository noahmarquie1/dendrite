from mesh_generation.snowflake_mesh import generate_full_snowflake
from particle_sim.solver import PointCloudSolver

N_BODIES = 100
DPI = 75

snowflake_mesh = generate_full_snowflake(1, 2, 3, 0.1)
width = snowflake_mesh.polygon.bounds[2] - snowflake_mesh.polygon.bounds[0]
height = snowflake_mesh.polygon.bounds[3] - snowflake_mesh.polygon.bounds[1]

solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    plots=None,
    polygon=snowflake_mesh.polygon,
)

sol = solver.solve(
    steps=int(1e3), 
)

solver.animate(out="./anim.gif")
