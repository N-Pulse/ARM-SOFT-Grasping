#!/usr/bin/env python3
"""
shape_id.py – V9
Detects  spherique | cylindrique | tablet | cuboid
Robust to partial / hollow / solid shapes
"""
import os, sys, json, struct, numpy as np

# ────────────────── I/O (.ply binary or JSON) ──────────────────────────────
def load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".ply":
        return np.asarray(json.load(open(path)), float)

    with open(path, "rb") as f:
        header, line = [], b""
        while not line.startswith(b"end_header"):
            line = f.readline()
            if not line:
                raise ValueError("PLY missing end_header")
            header.append(line)
        header = b"".join(header).decode(errors="ignore")

        # number of vertices and list of x,y,z types
        n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[2])
        coord_props = [l.split()[1] for l in header.splitlines() if l.startswith("property")][:3]
        has_color   = "uchar red" in header

        # dynamic struct format
        type_map = {"float": "f", "float32": "f", "double": "d", "float64": "d"}
        try:
            coord_fmt = "".join(type_map[t] for t in coord_props)
        except KeyError as t:
            raise ValueError(f"Type {t} not supported for coordinates")
        row_fmt  = "<" + coord_fmt + ("BBB" if has_color else "")
        row_size = struct.calcsize(row_fmt)

        raw = f.read(n * row_size)
        pts = np.array([struct.unpack_from(row_fmt, raw, i * row_size)[:3] for i in range(n)], float)
        return pts

# ────────────────── Geometry tools ─────────────────────────────────────────
def pca(pts):
    pts_c = pts - pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(pts_c, rowvar=False))
    idx = vals.argsort()[::-1]
    return vals[idx], vecs[:, idx], pts_c

def sphere_err(pts):
    A = np.c_[2 * pts, np.ones(len(pts))]
    f = (pts ** 2).sum(1)
    C, *_ = np.linalg.lstsq(A, f, rcond=None)
    c, c0 = C[:3], C[3]
    r = np.sqrt((c ** 2).sum() + c0)
    return np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - r) ** 2)) / r

def tube_stats(pts, axis):
    axis /= np.linalg.norm(axis)
    t      = pts @ axis
    radial = pts - np.outer(t, axis)
    radii  = np.linalg.norm(radial, axis=1)
    cv     = radii.std() / radii.mean()
    h2r    = t.ptp() / (2 * radii.mean())
    return cv, h2r

# ────────────────── Classification ─────────────────────────────────────────
def classify(points, *,
             sph_err_max=0.07,
             sph_iso_max=1.8,        # max λ1/λ2 for sphere
             tube_cv_max=0.50,
             cross_max=4.0,          # max λ2/λ3 for ≈ circular section
             tablet_max_h2r=0.6,
             debug=False):
    # PCA and common metrics
    (l1, l2, l3), vecs, pts_c = pca(points)
    r12 = l1 / (l2 + 1e-12)
    r23 = l2 / (l3 + 1e-12)

    # 1. Sphere – low error AND quasi-isotropy of the two main axes
    err = sphere_err(points)
    if debug:
        print(f"[sphere] err={err:.3f}  r12={r12:.2f}")
    if err < sph_err_max and r12 < sph_iso_max:
        return "spherique"

    # 2. Cylinder / Tablet – regular radius AND ~circular section
    cv, h2r = tube_stats(pts_c, vecs[:, 0])
    if debug:
        print(f"[tube] cv={cv:.3f} h/2r={h2r:.2f} r23={r23:.2f}")
    if cv < tube_cv_max and r23 < cross_max:
        return "tablet" if h2r < tablet_max_h2r else "cylindrique"

    # 3. Cuboid – fallback
    return "cuboid"

# ────────────────── Simple CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print("usage: python shape_id.py [-d] file.{ply|json}", file=sys.stderr)
        sys.exit(1)
    dbg  = "-d" in sys.argv
    file = sys.argv[-1]
    pts  = load_points(file)
    print(classify(pts, debug=dbg))
