import argparse
from importlib.metadata import version
from geometry.hex_geometry import Hexagon
from geometry.rect_geometry import Rect
from geometry.strict_mesh import StrictMesh
from geometry.base_geometry import extrude
from geometry.stats import Stats
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd


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
    ax.set_aspect(1)
    return fig, ax


def print_version():
    v = version("dendrite")
    print(f"Dendrite {v}.")


# Main CLI function
def generate(csv, static, step_size, out):
    if out:
        out = format_out_dir(out)
    else:
        out = "out/"

    shape_type = None

    try:
        df = pl.read_csv(csv)
        cols = df.columns
        if len(cols) == 4:
            shape_type = "hex"
        elif len(cols) == 5 and "radius" in cols:
            shape_type = "hex3d"
        elif len(cols) == 5:
            shape_type = "rect"
        elif len(cols) == 6:
            shape_type = "rect3d"
        else:
            raise ValueError
    except:
        print("Please provide valid shape via csv. Exiting ...")
        quit()

    shape_list = []
    depth = None
    for row in df.iter_rows():
        if shape_type == "hex":
            radius, trans_x, trans_y, theta = row
        elif shape_type == "hex3d":
            radius, depth, trans_x, trans_y, theta = row
        elif shape_type == "rect":
            width, height, trans_x, trans_y, theta = row
        else:
            width, height, depth, trans_x, trans_y, theta = row

        shape = Hexagon(radius, step_size) if shape_type == "hex" else Rect(width, height, step_size)
        shape.transform((trans_x, trans_y), theta)
        shape_list.append(shape)

    mesh = StrictMesh(shape_list, dynamic=not static)

    plt.style.use("seaborn-v0_8")
    fig, ax = reset_plots()
    plot_out = out + "plot.png"
    mesh.visualize(ax)
    plt.savefig(plot_out)
    print(f"Plot saved to {plot_out}.")

    if depth:
        z_max = depth / 2
        z_min = -depth / 2
        n_wall_steps = int((z_max - z_min) / step_size)

        top_face = np.column_stack([mesh.global_inner_points, np.full(len(mesh.global_inner_points), z_max)])
        bottom_face = np.column_stack([mesh.global_inner_points, np.full(len(mesh.global_inner_points), z_min)])
        walls = extrude(mesh.global_boundary_points, n_wall_steps, z_min, z_max)
        all_points = np.vstack([top_face, bottom_face, walls])

        points_df = pd.DataFrame(all_points)
        points_df.to_csv(
            "out/points.csv",
            header=False,
            index=False,
            float_format=lambda x: np.format_float_positional(x, trim='-'),
        )
        return

    else:
        all_points = np.vstack([
            mesh.global_boundary_points,
            mesh.global_inner_points,
        ])
        if not static:
            all_points = np.concatenate([all_points, np.vstack([dynamic_region.filled_points for dynamic_region in mesh.dynamic_regions])], axis=0)



    points_df = pd.DataFrame(all_points)
    points_df.to_csv(
        "out/points.csv",
        header=False,
        index=False,
         float_format=lambda x: np.format_float_positional(x, trim='-'),
    )

    stats = Stats(all_points, mesh.mesh, buffer=step_size*0.01)
    fig, ax = reset_plots()
    delaunay_out = out + "tri.png"
    stats.plot_delaunay(ax=ax)
    plt.savefig(delaunay_out)
    print(f"Delaunay Triangulation saved to {delaunay_out}.")

    plt.close('all')
    fig, ax = plt.subplots(1,1)
    pdf_out = out + "pdf.png"
    stats.plot_dists_pdf(ax)
    plt.savefig(pdf_out)
    print(f"PDF saved to {pdf_out}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true", help="Prints installed version of dendrite.")

    subparsers = parser.add_subparsers(dest="command", help="Available commands.")
    gen_parser = subparsers.add_parser("gen", help="Generate a mesh from provided polygons using a mix of static and dynamic methods.")

    gen_parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="Specify CSV file to create point distribution from."
    )

    gen_parser.add_argument(
        "--static",
        action="store_true",
        help="Does not generate dynamic point distributions if specified."
    )

    gen_parser.add_argument(
        "--step-size",
        required=True,
        type=float,
        help="Specifies ideal point spacing to be achieved."
    )

    gen_parser.add_argument(
        "--out",
        required=False,
        help="Specifies file to be written to. Uses 'out/' by defualt."
    )


    args = parser.parse_args()

    if args.version:
        print_version()

    elif args.command == "gen":
        generate(
            csv=args.csv,
            static=args.static,
            step_size=args.step_size,
            out=args.out,
        )
