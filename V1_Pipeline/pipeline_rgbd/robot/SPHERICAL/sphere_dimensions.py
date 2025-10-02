import numpy as np
import open3d as o3d

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

def classify_point_cloud(path: str):
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
    return center, radius, diameter, graspable, label
