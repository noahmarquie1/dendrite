import argparse
from importlib.metadata import version
from data.poly_management import load_polygon, save_polygon
from mesh_generation.mesh import Mesh
from mesh_generation.geometry import remove_in_area_points, extrude, plot_3d_element
from simulation.solver import PointCloudSolver
from shapely import Polygon
import numpy as np
import os
import matplotlib.pyplot as plt


# Constants - TEMPORARY
DPI = 75
N_BODIES = 100
FPS = 15
DENSITY = 100

# Helpers
def format_out_dir(out_dir):
    if out_dir is None:
        out_dir = "out/"
    os.makedirs(out_dir, exist_ok=True)
    if out_dir[-1] != "/":
        out_dir += "/"
    return out_dir


def reset_plots():
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    ax.set_aspect(1/1)
    return fig, ax


def print_version():
    v = version("dendrite")
    print(f"Dendrite {v}.")


def handle_pngs(out_dir):
    path = out_dir + "mesh.png"
    plt.savefig(path)
    print(f"2D mesh saved to {path}.")


def handle_csvs(out_dir, points):
    path = out_dir + "mesh.csv"
    print(f"Saving csv to {path}.")
    save_polygon(points, path)

def handle_extrude(out_dir, edge_points, inner_points):
    path = out_dir + "mesh_3d.png"
    plt.close()
    plot_3d_element(edge_points, inner_points)
    plt.savefig(path)
    print(f"3D Mesh saved to {path}")
    

# Main CLI function
def generate(static_poly, dynamic_poly, anim, extrude, png, csv, out_dir):
    out_dir = format_out_dir(out_dir)
    fig, ax = plt.subplots(1,1)
    ax.set_aspect("equal")

    if static_poly:
        s_vertices = load_polygon(static_poly)
        s_mesh = Mesh(Polygon(s_vertices))
        edge_points = s_mesh.edge_points
        inner_points = s_mesh.inner_points
        total_points = np.vstack([s_mesh.inner_points, s_mesh.edge_points])
        if not dynamic_poly:
            if anim:
                print("Cannot generate animation for static mesh. Proceeding ...")
            if csv:
                print(f"Saving csv to {csv}.")
                handle_csvs(csv, total_points)
            if png:
                print(f"Saving PNG to {png}.")
                s_mesh.visualize(ax=ax)
                handle_pngs(png)
    
    if dynamic_poly:
        anim_color = 'blue'
        d_vertices = load_polygon(dynamic_poly)
        dynamic_poly = Polygon(d_vertices)
        n_bodies = int(DENSITY * dynamic_poly.area)
        solver = PointCloudSolver(dpi=DPI, plots=None, n_bodies=n_bodies, polygon=dynamic_poly, fps=FPS)
        sol = solver.solve(steps=int(1e4))

        if static_poly:
            s_mesh.add_shape(Polygon(d_vertices))
            s_mesh.inner_points = remove_in_area_points(s_mesh.inner_points, d_vertices)
            s_mesh.inner_points = np.vstack([s_mesh.inner_points, s_mesh.edge_fill(Polygon(d_vertices))])
            solver.anim.add_static_points(s_mesh.inner_points, "red")
            solver.anim.add_static_points(s_mesh.edge_points, "blue")
            fig, ax = reset_plots()
            anim_color = 'purple'
            edge_points = s_mesh.edge_points
            inner_points = np.vstack([sol[-1][:n_bodies], s_mesh.inner_points])
            total_points = np.vstack([edge_points, inner_points])
            s_mesh.visualize(ax=ax)
            ax.scatter(sol[-1][:n_bodies, 0], sol[-1][:n_bodies, 1], s=4, c='purple')
        else:
            fig, ax = reset_plots()
            d_mesh = Mesh(Polygon(d_vertices))
            edge_points = d_mesh.edge_points
            inner_points = sol[-1][:n_bodies]
            total_points = np.vstack([edge_points, inner_points])
            ax.scatter(total_points[:, 0], total_points[:, 1])

        if png:
            handle_pngs(out_dir)
        if anim:
            solver.animate(out=out_dir + "anim.gif", color=anim_color)
        if csv:
            handle_csvs(out_dir, total_points)
        if extrude:
            handle_extrude(out_dir, edge_points, inner_points)
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
        action="store_true",
        help="Boolean flag to include animations. Can be 'gif' or 'mp4.'"
    )
    gen_parser.add_argument(
        "--extrude", 
        required=False,
        action="store_true",
        help="Boolean flag to include 3D mesh."
    )
    gen_parser.add_argument(
        "--png", 
        required=False, 
        action="store_true",
        help="Specifies directory for outputted PNGs of 2D mesh and/or extruded 3D mesh."
    )
    gen_parser.add_argument(
        "--csv", 
        required=False, 
        action="store_true",
        help="Specifies location for CSV of final mesh."
    )
    gen_parser.add_argument(
        "--out-dir",
        required=False,
        type=str,
        help="Specifies directory for output data and images."
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
            out_dir=args.out_dir,
        )
