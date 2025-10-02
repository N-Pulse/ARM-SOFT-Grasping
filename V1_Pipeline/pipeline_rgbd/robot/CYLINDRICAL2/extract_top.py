import open3d as o3d
import numpy as np
import argparse
import os
import sys

from pca import align_point_cloud_to_z_axis 
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SLICE_HEIGHT = 0.005  # meters (1 cm at the top of the aligned cloud c'est frér
# -----------------------------------------------------------------------------
def extract_top_slice(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Given a Z-aligned point cloud `pcd`, return all points whose
    Z coordinate lies within SLICE_HEIGHT of the maximum Z.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Point cloud has no points!")
    max_z = float(np.max(pts[:, 2]))
    threshold = max_z - SLICE_HEIGHT
    mask = pts[:, 2] >= threshold
    indices = np.where(mask)[0].tolist()
    return pcd.select_by_index(indices)

# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Align via PCA and extract the top SLICE_HEIGHT meters of a point cloud."
    )
    parser.add_argument("input", help="Input point cloud file (PLY/PCD/etc.)")
    parser.add_argument(
        "-o", "--output",
        help="Output filename for the top slice; defaults to '<input_basename>_top_slice<ext>'."
    )
    return parser.parse_args()

# -----------------------------------------------------------------------------
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

    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        print("Error: Loaded point cloud is empty!", file=sys.stderr)
        sys.exit(1)

    # Align the cloud to Z axis
    aligned = align_point_cloud_to_z_axis(pcd)

    # Extract and save the top slice
    top_slice = extract_top_slice(aligned)

    if args.output:
        output_name = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_name = f"{base}_top_slice{ext}"
    output_path = os.path.join(script_dir, output_name)

    if not o3d.io.write_point_cloud(output_path, top_slice):
        print(f"Error: Failed to write '{output_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Saved top slice ({len(top_slice.points)} points) to '{output_path}'.")

if __name__ == "__main__":
    main()
