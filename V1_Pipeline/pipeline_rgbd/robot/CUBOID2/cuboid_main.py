#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuboid_main.py simplifié - Plus de sélection ArUco
Extrait directement les dimensions et effectue le grasp
Ajout de visualisations pour le nuage et les dimensions
"""

import os
import sys
import re
import subprocess
import argparse
import cv2
import numpy as np
import open3d as o3d
from cuboid_dimensions import build_obb

# ─── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ─── Argument pour le .ply ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simplified cuboid grasp pipeline")
parser.add_argument(
    "ply",
    help="Chemin vers le fichier point-cloud (.ply) généré par la pipeline",
)
args = parser.parse_args()

# ─── Helpers ──────────────────────────────────────────────────────────────

def extract_dimensions_from_text(text):
    """
    Parse lines like "Length : 0.0580" → returns dict of dimensions and (width, length, height).
    """
    dims = {k.lower(): float(v) for k, v in re.findall(r"(\w+)\s*:\s*([\d\.]+)", text)}
    return dims, (dims.get("width", 0.0), dims.get("length", 0.0), dims.get("height", 0.0))


def show_dimensions_summary(width: float, length: float, height: float) -> None:
    """Affiche un récapitulatif graphique des dimensions."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Cuboid dimensions (m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Width : {width:.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img, f"Length: {length:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"Height: {height:.3f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    max_dim = max(width, length, height, 0.1)
    bar_width = 200
    cv2.rectangle(img, (300, 70), (300 + int(bar_width * width / max_dim), 90), (255, 0, 0), -1)
    cv2.rectangle(img, (300, 110), (300 + int(bar_width * length / max_dim), 130), (0, 255, 0), -1)
    cv2.rectangle(img, (300, 150), (300 + int(bar_width * height / max_dim), 170), (0, 0, 255), -1)
    cv2.imshow("Dimensions Recap", img)
    cv2.waitKey(0)

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"[CUBOID] Traitement du fichier : {args.ply}")
    
    # 1) Extraire les dimensions du nuage de points
    try:
        print("[CUBOID] Extraction des dimensions...")
        dims_out = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "cuboid_dimensions.py"), args.ply, "--no-visualize"],
            capture_output=True, text=True, check=True
        ).stdout
        
        dims_named, (w, l, h) = extract_dimensions_from_text(dims_out)
        print(f"[CUBOID] Dimensions extraites : width={w:.3f}m, length={l:.3f}m, height={h:.3f}m")
        
        # Visualisation du nuage avec OBB
        print("[CUBOID] Visualisation du nuage d'entrée avec OBB...")
        pcd = o3d.io.read_point_cloud(args.ply)
        axes = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]
        obb = build_obb(np.mean(np.asarray(pcd.points), axis=0), axes, dims_named)
        obb.color = (1.0, 0.0, 0.0)
        pcd_copy = o3d.geometry.PointCloud(pcd)
        pcd_copy.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([pcd_copy, obb], window_name="Nuage d'entrée avec OBB", width=900, height=600)

        # Récapitulatif graphique des dimensions
        show_dimensions_summary(w, l, h)
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec de l'extraction des dimensions : {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # 2) Effectuer le grasp avec la longueur
    try:
        print(f"[CUBOID] Exécution du grasp avec length={l:.3f}m...")
        grasp_cmd = [
            sys.executable, 
            os.path.join(SCRIPT_DIR, "cuboid_grasp.py"), 
            "--length", f"{l:.3f}"
        ]
        
        result = subprocess.run(grasp_cmd, capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print(f"[CUBOID] {result.stdout.strip()}")
        
        print("[CUBOID] ✅ Séquence de grasp terminée avec succès")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Échec du grasp : {e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

    # ── Attente de la touche 'r' pour remise à zéro ─────────────────────────
    print("Appuyez sur 'r' pour remettre le bras à zéro, ou toute autre touche pour quitter.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):
        reset_path = os.path.join(SCRIPT_DIR, os.pardir, "reset.py")
        if not os.path.exists(reset_path):
            print(f"[ERR] Le fichier '{reset_path}' n'existe pas.", file=sys.stderr)
            sys.exit(1)
        print(f"[CUBOID] Lancement de {reset_path} …")
        ret2 = subprocess.run([sys.executable, reset_path], check=False)
        if ret2.returncode != 0:
            print(f"[ERR] reset.py s'est terminé avec le code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("[CUBOID] Fin du programme.")

if __name__ == "__main__":
    main()