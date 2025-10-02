import open3d as o3d
import numpy as np
from scipy.optimize import least_squares
import argparse, os, sys

def generate_initial_guesses(points):
    """
    Generate initial (centroid, axis, radius) guesses using the three PCA directions.
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    guesses = []
    for idx in range(3):
        u0 = Vt[idx] / np.linalg.norm(Vt[idx])
        d = np.linalg.norm(np.cross(points - centroid, u0), axis=1)
        r0 = d.mean()
        guesses.append((centroid, u0, r0))
    return guesses

def residuals(x, points):
    c = x[0:3]
    u = x[3:6]; u = u/np.linalg.norm(u)
    r = x[6]
    d = np.linalg.norm(np.cross(points - c, u), axis=1)
    return d - r

def fit_cylinder(points):
    best_cost = np.inf
    best_sol = None
    for c0, u0, r0 in generate_initial_guesses(points):
        x0 = np.hstack((c0, u0, r0))
        try:
            res = least_squares(residuals, x0, args=(points,), xtol=1e-8, ftol=1e-8)
        except Exception:
            continue
        if not res.success:
            continue
        cost = np.sum(res.fun**2)
        if cost < best_cost:
            best_cost = cost
            x_opt = res.x
            c_opt = x_opt[0:3]
            u_opt = x_opt[3:6] / np.linalg.norm(x_opt[3:6])
            r_opt = x_opt[6]
            best_sol = (c_opt, u_opt, r_opt)
    if best_sol is None:
        raise RuntimeError("Cylinder fit failed for all initializations.")
    return best_sol

def build_cylinder_mesh(c, u, r, points, resolution=64):
    d_proj = (points - c).dot(u)
    d_min, d_max = d_proj.min(), d_proj.max()
    length = d_max - d_min
    midpoint = (d_min + d_max)/2.0
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=length, resolution=resolution)
    mesh.compute_vertex_normals()
    z = np.array([0,0,1.0])
    v = np.cross(z, u)
    if np.linalg.norm(v)>1e-6:
        angle = np.arccos(np.dot(z,u))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(v/np.linalg.norm(v)*angle)
        mesh.rotate(R, center=(0,0,0))
    mesh.translate(c + u*midpoint)
    return mesh

def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("input", help="Input point cloud file in same folder")
    p.add_argument("--save-cylinder", metavar="CYL", help="Save cylinder mesh path")
    p.add_argument("--save-merged", metavar="MERGED", help="Save merged cloud path")
    p.add_argument("--samples", type=int, default=100000,
                   help="# points for merged cloud sampling")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"Error: '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(args.input)
    if pcd.is_empty():
        print("Error: loaded cloud is empty.", file=sys.stderr)
        sys.exit(1)
    pts = np.asarray(pcd.points)
    print("Fitting cylinder (multi-start PCA + LS)...")
    c, u, r = fit_cylinder(pts)
    d_proj = (pts - c).dot(u)
    height = d_proj.max() - d_proj.min()
    diameter = 2*r
    print(f" • Axis point (c):     {c}")
    print(f" • Axis direction (u): {u}")
    print(f" • Radius (r):         {r:.6f}")
    print(f" • Height (h):         {height:.6f}")
    print(f" • Diameter:           {diameter:.6f}")
    cyl_mesh = build_cylinder_mesh(c, u, r, pts)
    if args.save_cylinder:
        o3d.io.write_triangle_mesh(args.save_cylinder, cyl_mesh)
        print(f"Saved cylinder mesh to '{args.save_cylinder}'")
    if args.save_merged:
        cyl_pcd = cyl_mesh.sample_points_uniformly(number_of_points=args.samples)
        cyl_pcd.paint_uniform_color([1,0,0])
        pcd.paint_uniform_color([0.6,0.6,0.6])
        merged = pcd + cyl_pcd
        o3d.io.write_point_cloud(args.save_merged, merged)
        print(f"Saved merged cloud to '{args.save_merged}'")
    print("Visualization skipped for seamless execution.")

if __name__=="__main__":
    main()