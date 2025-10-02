from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

from sphere_dimensions import classify_point_cloud

def show_parameters_summary(diameter: float, graspable: bool, label: str) -> None:
    """Affiche un récapitulatif graphique des paramètres."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Sphere Parameters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Diameter : {diameter:.4f} m", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img, f"Prehensible : {graspable}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"Type of grasp : {label}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Parameters Recap", img)
    cv2.waitKey(0)

def main() -> None:
    """Point d'entrée principal du pipeline de préhension sphérique."""
    script_dir = Path(__file__).resolve().parent
    default_pc = script_dir / "sphere.ply"

    parser = argparse.ArgumentParser(
        description="Détermine le type de prise pour un objet sphérique et appelle le contrôleur de robot."
    )
    parser.add_argument(
        "input_pc",
        nargs="?",
        default=str(default_pc),
        help=f"Chemin vers le nuage de points filtré (défaut : '{default_pc}').",
    )
    args = parser.parse_args()

    pc_path = Path(args.input_pc)
    if not pc_path.exists():
        parser.error(f"Le fichier point‑cloud '{pc_path}' n'existe pas.")

    # Analyse du nuage de points
    center, radius, diameter, graspable, label = classify_point_cloud(str(pc_path))
    print(f"Diamètre calculé : {diameter:.4f} m")
    print(f"Préhensible : {graspable} (label : {label})")

    # Visualisation du nuage d'entrée avec sphère
    print("Visualisation du nuage d'entrée avec sphère ajustée...")
    pcd = o3d.io.read_point_cloud(str(pc_path))
    pcd_copy = o3d.geometry.PointCloud(pcd)
    pcd_copy.paint_uniform_color([0.6, 0.6, 0.6])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.translate(center)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    center_marker.translate(center)
    center_marker.paint_uniform_color([0.0, 1.0, 0.0])
    o3d.visualization.draw_geometries(
        [pcd_copy, sphere, center_marker],
        window_name="Nuage d'entrée avec sphère",
        width=900,
        height=600
    )

    # Récapitulatif graphique
    show_parameters_summary(diameter, graspable, label)

    # Type de prise
    grasp = label

    # Chemin absolu vers sphere_grasp.py
    grasp_script = script_dir / "sphere_grasp.py"
    if not grasp_script.exists():
        print(f"Erreur : Le fichier '{grasp_script}' n'existe pas.", file=sys.stderr)
        sys.exit(1)

    # Appel du script de prise
    cmd = [
        sys.executable,
        str(grasp_script),
        "--grasp",
        grasp,
        "--distance",
        f"{diameter:.6f}",
    ]
    print("Exécution :", " ".join(cmd))

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"Erreur : sphere_grasp.py s'est terminé avec le code {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

    # Ferme les fenêtres OpenCV
    cv2.destroyAllWindows()

    # ── Attente de la touche 'r' pour reset ─────────────────────────────
    print("Appuyez sur 'r' pour remettre le bras à zéro, ou toute autre touche pour quitter.")
    # attend une touche dans une fenêtre nommée (inutile d'afficher la fenêtre)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):
        reset_path = script_dir.parent / "reset.py"
        if not reset_path.exists():
            print(f"Erreur : Le fichier '{reset_path}' n'existe pas.", file=sys.stderr)
            sys.exit(1)
        print(f"Lancement de {reset_path} …")
        ret2 = subprocess.run([sys.executable, str(reset_path)], check=False)
        if ret2.returncode != 0:
            print(f"Erreur : reset.py s'est terminé avec le code {ret2.returncode}", file=sys.stderr)
            sys.exit(ret2.returncode)
    else:
        print("Fin du programme.")

if __name__ == "__main__":
    main()
