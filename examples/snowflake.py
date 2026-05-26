from particle_sim.solver import PointCloudSolver
from mesh_generation.mesh import MultiMesh

N_BODIES = int(1e3)
DPI = 75

snowflake_mesh = MultiMesh(dir="data/polygons/snowflake/")
solver = PointCloudSolver(
    dpi=DPI,
    n_bodies=N_BODIES,
    plots=None,
    polygon=snowflake_mesh.polygon,
    width=6,
    height=6,
)

sol = solver.solve(
    steps=int(1e4), 
)

solver.animate(out="./anim.gif")
