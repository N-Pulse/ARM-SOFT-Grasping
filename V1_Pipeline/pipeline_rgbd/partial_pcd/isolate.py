import argparse
import sys
import numpy as np
import open3d as o3d

def isolate_object(pcd, voxel, plane_d, eps, min_pts, keep_normals):
    # 1. (Optional) Estimate normals if requested
    if keep_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

    # 2. Voxel down-sampling
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    
    # 3. Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 4. Remove dominant plane (e.g., table/floor)
    plane_model, inliers = pcd.segment_plane(plane_d, 3, 1000)
    pcd = pcd.select_by_index(inliers, invert=True)

    # 5. Clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts))
    valid = labels[labels >= 0]
    if valid.size == 0:
        raise RuntimeError("No cluster detected; decrease min_points or increase eps.")
    largest = np.bincount(valid).argmax()
    return pcd.select_by_index(np.where(labels == largest)[0])

# ------------------------------------------------------------------
# Usage: python isoler_ply.py scene.ply --voxel 0.001 --min_pts 10
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Isolate main object from a .ply point cloud")
parser.add_argument("ply", help="input .ply file")
parser.add_argument("-o", "--out", default="isolated_object.ply", help="output .ply file")
parser.add_argument("--voxel",    type=float, default=0.001, help="voxel size for down-sampling (0 = no down-sampling, in meters)")
parser.add_argument("--plane_d",  type=float, default=0.004, help="RANSAC plane distance threshold (meters)")
parser.add_argument("--eps",      type=float, default=0.015, help="DBSCAN radius (meters)")
parser.add_argument("--min_pts",  type=int,   default=15,    help="minimum points for DBSCAN cluster")
parser.add_argument("--keep_normals", action="store_true", help="estimate and keep normals")
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.ply)
try:
    obj = isolate_object(pcd, args.voxel, args.plane_d, args.eps, args.min_pts, args.keep_normals)
except Exception as e:
    sys.exit(f"❌  {e}")

o3d.io.write_point_cloud(args.out, obj)
print(f"✅  {args.out}  ({len(obj.points)} pts)")
o3d.visualization.draw_geometries([obj], window_name="Isolated Object")
