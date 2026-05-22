import argparse
from importlib.metadata import version
from data.poly_management import load_polygon, save_polygon
from mesh_generation.mesh import Mesh
from mesh_generation.geometry import remove_in_area_points
from particle_sim.solver import PointCloudSolver
from shapely import Polygon
import numpy as np
import os
import matplotlib.pyplot as plt


# Constants - TEMPORARY
DPI = 75
N_BODIES = 100
DRAG_COEF = 100
FPS = 15
FORCE_MULTIPLIER = 1

T = 6
step = T * 1e-2

def reset_plots():
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    ax.set_aspect(1/1)
    return fig, ax



def print_version():
    v = version("dendrite")
    print(f"Dendrite {v}.")


def handle_pngs(path):
    os.makedirs(path, exist_ok=True)
    if path[-1] == "/":
        path = path[:-1]
        path += "/mesh.png"
    print(f"Saving 2D PNG to {path}.")
    plt.savefig(path)


def handle_csvs(path, points):
    if path[-1] == '/':
        path += "mesh.csv"
    print(f"Saving csv to {path}.")
    save_polygon(points, path)
    


def generate(static_poly, dynamic_poly, anim, extrude, png, csv):
    fig, ax = plt.subplots(1,1)
    ax.set_aspect("equal")

    if static_poly:
        s_vertices = load_polygon(static_poly)
        s_mesh = Mesh(Polygon(s_vertices))
        if not dynamic_poly:
            if anim:
                print("Cannot generate animation for static mesh. Proceeding ...")
            if csv:
                print(f"Saving csv to {csv}.")
                final_points = np.vstack([s_mesh.inner_points, s_mesh.edge_points])
                handle_csvs(csv, final_points)
            if png:
                print(f"Saving PNG to {png}.")
                s_mesh.visualize(ax=ax)
                handle_pngs(png)
            return
    
    if dynamic_poly:
        anim_color = 'blue'
        d_vertices = load_polygon(dynamic_poly)
        solver = PointCloudSolver(
            dpi=DPI,
            n_bodies=N_BODIES,
            force_multiplier=FORCE_MULTIPLIER,
            width=6,
            height=6,
            drag_coeff=DRAG_COEF,
            plots=None,
            polygon=Polygon(d_vertices),
            fps=FPS,
        )
        sol = solver.solve(
            max_step=step, 
            steps=int(1e3), 
        )

        final_points = None
        if static_poly:
            s_mesh.add_shape(Polygon(d_vertices))
            s_mesh.inner_points = remove_in_area_points(s_mesh.inner_points, d_vertices)
            s_mesh.inner_points = np.vstack([s_mesh.inner_points, s_mesh.edge_fill(Polygon(d_vertices))])
            solver.anim.add_static_points(s_mesh.inner_points, "red")
            solver.anim.add_static_points(s_mesh.edge_points, "blue")
            fig, ax = reset_plots()
            anim_color = 'purple'
            final_points = np.vstack([sol[-1][:N_BODIES], s_mesh.inner_points, s_mesh.edge_points])
            s_mesh.visualize(ax=ax)
            ax.scatter(sol[-1][:N_BODIES, 0], sol[-1][:N_BODIES, 1], s=4, c='purple')
        else:
            fig, ax = reset_plots()
            d_mesh = Mesh(Polygon(d_vertices))
            final_points = np.vstack([d_mesh.edge_points, sol[-1][:N_BODIES]])
            ax.scatter(final_points[:, 0], final_points[:, 1])

        if png:
            handle_pngs(png)
        if anim:
            solver.animate(out=anim, color=anim_color)
        if csv:
            handle_csvs(csv, final_points)
        
        return

    print("Please specify either a static or dynamic polygon for mesh creation. Quitting ...")
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true", help="Prints installed version of dendrite.")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands.")
    gen_parser = subparsers.add_parser("gen", help="Generate a mesh from provided polygons using a mix of static and dynamic methods.")

    gen_parser.add_argument(
        "--static-poly", 
        required=False, 
        type=str, 
        help="Specify polygon path (csv) to be filled in statically."
    )
    gen_parser.add_argument(
        "--dynamic-poly", 
        required=False, 
        type=str, 
        help="Specify polygon path (csv) to be filled in dynamically."
    )
    gen_parser.add_argument(
        "--anim", 
        required=False, 
        type=str,
        help="Boolean flag to include animations. Can be 'gif' or 'mp4.'"
    )
    gen_parser.add_argument(
        "--extrude", 
        required=False, 
        help="Boolean flag to include 3D mesh."
    )
    gen_parser.add_argument(
        "--png", 
        required=False, 
        type=str,
        help="Specifies directory for outputted PNGs of 2D mesh and/or extruded 3D mesh."
    )
    gen_parser.add_argument(
        "--csv", 
        required=False, 
        type=str,
        help="Specifies location for CSV of final mesh."
    )

    args = parser.parse_args()

    if args.version:
        print_version()

    elif args.command == "gen":
        generate(
            static_poly=args.static_poly, 
            dynamic_poly=args.dynamic_poly,
            anim=args.anim, 
            extrude=args.extrude,
            png=args.png,
            csv=args.csv,
        )
