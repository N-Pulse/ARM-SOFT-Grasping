import open3d as o3d
import numpy as np

def align_point_cloud_to_z_axis(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Aligns the input point cloud so that its principal component (largest variance direction)
    is aligned with the global Z axis and centers the cloud at the origin. Handles partial clouds by
    ensuring a consistent 'up' direction.

    Args:
        pcd: Open3D PointCloud object

    Returns:
        A new Open3D PointCloud aligned with Z axis and centered at (0,0,0).
    """
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    cov = np.cov(pts_centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]
    principal = eig_vecs[:, 0]

    # Ensure the principal axis points upward (positive Z)
    if principal[2] < 0:
        principal = -principal

    # Compute rotation to align principal axis to Z-axis
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(principal, z)
    s = np.linalg.norm(v)
    if s < 1e-8:
        R = np.eye(3)
    else:
        c = np.dot(principal, z)
        vx = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],    0]])
        R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))

    pts_aligned = pts_centered.dot(R.T)

    # Flip Z if mean height is negative
    if pts_aligned[:, 2].mean() < 0:
        pts_aligned[:, 2] *= -1
        R = np.diag([1, 1, -1]).dot(R)

    aligned = o3d.geometry.PointCloud()
    aligned.points = o3d.utility.Vector3dVector(pts_aligned)
    if pcd.has_colors():
        aligned.colors = pcd.colors
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        aligned.normals = o3d.utility.Vector3dVector(normals.dot(R.T))
    return aligned


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align and visualize a partial point cloud via PCA, with axes and origin.")
    parser.add_argument('-i', '--input', required=True, help='Input point cloud file (PLY, PCD)')
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    if not pcd.has_points():
        print('Error: empty or invalid point cloud.')
        return

    aligned = align_point_cloud_to_z_axis(pcd)

    # Shift original to left; leave aligned at origin
    shift = np.max(np.abs(np.asarray(pcd.points))[:, 0]) + 0.1
    orig_vis = pcd.translate((-shift, 0, 0), relative=True)
    aligned_vis = aligned  # already centered at (0,0,0)

    # Create coordinate axes and origin marker at world origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    origin_marker.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([
        orig_vis.paint_uniform_color([1, 0.706, 0]),
        aligned_vis.paint_uniform_color([0, 0.651, 0.929]),
        axes,
        origin_marker
    ],
    window_name='Original (left) vs Aligned (center) with Axes',
    width=800, height=600)

if __name__ == '__main__':
    main()
