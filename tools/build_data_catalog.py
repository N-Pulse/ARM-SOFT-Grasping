#!/usr/bin/env python3
"""Build a small, reproducible index of the shared ARM-SOFT data root."""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from armsoft.data import data_root

def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=data_root())
    args=parser.parse_args(); root=args.data_root.resolve(); sessions=[]
    capture_root=root/"raw"/"object"
    for path in sorted(capture_root.glob("*")):
        metadata_path=path/"session.json"
        if not metadata_path.is_file(): continue
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        annotations=root/"annotations"/"object_segmentation"/path.name/"annotations"
        states={"approved":0,"rejected":0,"pending":0}
        if annotations.exists():
            for item in annotations.glob("*.json"):
                try: state=json.loads(item.read_text()).get("review_status","pending")
                except (OSError,json.JSONDecodeError): continue
                states[state]=states.get(state,0)+1
        derived=root/"derived"/"object"/path.name
        annotation_run=root/"annotations"/"object_segmentation"/path.name/"annotation_run.json"
        active_preprocessing_version=None
        if annotation_run.is_file():
            try:
                run=json.loads(annotation_run.read_text(encoding="utf-8"))
                active_preprocessing_version=run.get("preprocessing_version") or Path(run.get("derived_masks","")).name or None
            except (OSError,json.JSONDecodeError): pass
        sessions.append({"id":path.name,"purpose":metadata.get("purpose"),
                         "created_utc":metadata.get("created_utc"),
                         "frame_count":metadata.get("frame_count",0),
                         "raw_bytes":size_bytes(path),"derived_versions":sorted(p.name for p in derived.glob("*") if p.is_dir()),
                         "active_preprocessing_version":active_preprocessing_version,
                         "annotation_states":states})
    result={"schema":"armsoft.catalog.v1","generated_utc":datetime.now(timezone.utc).isoformat(),
            "data_root":str(root),"sessions":sessions}
    destination=root/"catalog"/"index.json"; destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"[catalog] indexed {len(sessions)} session(s) -> {destination}")
    return 0
if __name__=="__main__": raise SystemExit(main())
