import open3d as o3d
import numpy as np
import argparse
import os
import sys
from pca import align_point_cloud_to_z_axis

# Configuration: half the slice thickness in meters (e.g. 0.01 = 1 cm)
SLICE_HALF_HEIGHT = 0.01
# Configuration: proportional offset to apply to the Z midpoint.
PROP_Z_OFFSET = 0.0

def extract_body_slice(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Extract a slice of thickness 2 * SLICE_HALF_HEIGHT around Z = 0 of a centered 'pcd'.
    Assumes the cloud has been translated so its Z-midpoint lies at the origin.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Point cloud has no points!")
    lower, upper = -SLICE_HALF_HEIGHT, SLICE_HALF_HEIGHT
    mask = (pts[:, 2] >= lower) & (pts[:, 2] <= upper)
    indices = np.where(mask)[0].tolist()
    return pcd.select_by_index(indices)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aligns a partial cylindrical point cloud via PCA, centers by Z-midpoint (with proportional offset), and extracts a configurable central slice"
    )
    parser.add_argument(
        "input",
        help="Filename of the input point cloud (PLY/PCD/XYZ etc.) in the same directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Filename to save the extracted body slice; defaults to '<input_basename>_body_slice<ext>' in the same directory"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.input):
        input_path = args.input
    else:
        input_path = os.path.join(script_dir, args.input)
    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading point cloud from '{input_path}'…")
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        print("Error: Loaded point cloud is empty!", file=sys.stderr)
        sys.exit(1)

    aligned = align_point_cloud_to_z_axis(pcd)
    pts = np.asarray(aligned.points)
    min_z, max_z = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))
    total_height = max_z - min_z
    mid_z = (min_z + max_z) / 2.0 - PROP_Z_OFFSET * total_height
    print(f"min_z = {min_z:.4f}, max_z = {max_z:.4f}, total_height = {total_height:.4f}")
    print(f"Computed midpoint = {(min_z + max_z) / 2.0:.4f}, offset = {PROP_Z_OFFSET} * height = {PROP_Z_OFFSET * total_height:.4f}, final mid_z = {mid_z:.4f}")
    aligned.translate((0.0, 0.0, -mid_z))
    print(f"Translated cloud by -{mid_z:.4f} in Z to center midpoint at origin")

    print("Extracting central slice around Z=0…")
    body_slice = extract_body_slice(aligned)

    if args.output:
        output_name = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_name = f"{base}_body_slice{ext}"
    output_path = os.path.join(script_dir, output_name)

    print(f"Saving extracted slice to '{output_path}'…")
    success = o3d.io.write_point_cloud(output_path, body_slice)
    if not success:
        print(f"Error: Failed to write '{output_path}'", file=sys.stderr)
        sys.exit(1)
    print("Save complete.")
    print("Visualization skipped for seamless execution.")

if __name__ == "__main__":
    main()