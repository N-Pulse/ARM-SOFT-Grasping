import numpy as np
import open3d as o3d
import argparse

# Hard-coded graspable diameter
MAX_GRASPABLE_DIAMETER = 0.20

def fit_sphere(pts: np.ndarray):
    """
    Fit a sphere to an Nx3 array of points via linear least-squares.
    Returns (center, radius).
    """
    X, Y, Z = pts[:,0], pts[:,1], pts[:,2]
    A = np.column_stack([X, Y, Z, np.ones_like(X)])  # (N,4)
    b = -(X**2 + Y**2 + Z**2)
    D, E, F, G = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy, cz = -D/2.0, -E/2.0, -F/2.0
    r = np.sqrt(cx*cx + cy*cy + cz*cz - G)
    return (cx, cy, cz), r

def classify_point_cloud(path: str, visualize: bool = False):
    """
    Load point cloud from 'path', fit sphere, compute diameter,
    and decide graspability.
    Returns:
      center: (x,y,z)
      radius: float
      diameter: float
      graspable: bool
      label: "spherical" or "flat"
    """
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"No points loaded from '{path}'")
    center, radius = fit_sphere(pts)
    diameter = 2.0 * radius
    graspable = (diameter <= MAX_GRASPABLE_DIAMETER)
    label = "spherical" if graspable else "flat"

    if visualize:
        print("Visualisation du nuage de points avec la sphère ajustée...")
        # Colorer le nuage en gris
        pcd_copy = o3d.geometry.PointCloud(pcd)
        pcd_copy.paint_uniform_color([0.6, 0.6, 0.6])
        # Créer une sphère au centre avec le rayon ajusté
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Rouge
        # Ajouter un marqueur pour le centre
        center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        center_marker.translate(center)
        center_marker.paint_uniform_color([0.0, 1.0, 0.0])  # Vert
        o3d.visualization.draw_geometries(
            [pcd_copy, sphere, center_marker],
            window_name="Nuage avec sphère ajustée",
            width=900,
            height=600
        )

    return center, radius, diameter, graspable, label

def main():
    parser = argparse.ArgumentParser(description="Analyse un nuage de points sphérique et affiche les résultats.")
    parser.add_argument("path", help="Chemin vers le fichier PLY/PCD")
    parser.add_argument("--visualize", action="store_true", help="Affiche la visualisation du nuage et de la sphère")
    args = parser.parse_args()

    center, radius, diameter, graspable, label = classify_point_cloud(args.path, visualize=args.visualize)
    print(f"Centre : {center}")
    print(f"Rayon : {radius:.4f} m")
    print(f"Diamètre : {diameter:.4f} m")
    print(f"Préhensible : {graspable} (label : {label})")

if __name__ == "__main__":
    main()