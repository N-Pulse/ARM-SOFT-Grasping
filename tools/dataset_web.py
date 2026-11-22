#!/usr/bin/env python3
"""Local browser UI for previewing and collecting RGB-D datasets."""
from __future__ import annotations
import argparse, csv, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from armsoft.data import CaptureSessionWriter, SESSION_SCHEMA_VERSION, data_root
from armsoft.models import discover_runtime_models
from armsoft.object_segmentation.annotations import FrameAnnotation, ObjectAnnotation
from armsoft.object_segmentation.export import to_yolo_lines
from armsoft.perception import adapt_ultralytics_result
from armsoft.perception import box_core_depth as perception_box_core_depth
from armsoft.perception import deproject_pixel
from armsoft.pointcloud import PointCloudConfig, PointCloudPipeline, save_ply
from armsoft.preprocessing import DepthFilterConfig, apply_depth_filter
from armsoft.sources import create_frame_source
from armsoft.sources.base import RGBDFrame
from armsoft.studio import create_studio_handler

ROOT=Path(__file__).resolve().parents[1]; WEB=ROOT/"web/dataset-console"; SAFE=re.compile(r"^(?!\.{1,2}$)(?!\.)[\w.-]{1,80}$")
MAX_REQUEST_BYTES=1024*1024
def sudo_owner():
    """Return the invoking Unix user IDs, or None on Windows/non-sudo runs."""
    if os.name!="posix" or not hasattr(os,"geteuid") or os.geteuid()!=0 or not os.environ.get("SUDO_UID"): return None
    uid=int(os.environ["SUDO_UID"]); return uid,int(os.environ.get("SUDO_GID",uid))

def compute_devices(recommended=False):
    devices=[{"id":"cpu","label":"CPU (compatible, slower)" if recommended else "CPU"}]
    try:
        import torch
        if torch.cuda.is_available(): devices.insert(0,{"id":"cuda:0","label":"NVIDIA GPU / CUDA (recommended)"})
        elif sys.platform=="darwin" and torch.backends.mps.is_available(): devices.insert(0,{"id":"mps","label":"Apple GPU / MPS (recommended)" if recommended else "Apple GPU / MPS"})
    except (ImportError,AttributeError): pass
    return devices

def jpeg(im):
    ok, data=cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY,86])
    if not ok: raise RuntimeError("JPEG encoding failed")
    return data.tobytes()

def box_core_depth(depth,box,fraction=.5):
    """Compatibility wrapper for the shared perception depth fusion."""
    return perception_box_core_depth(depth,box,fraction)

def live_object(names,boxes,index,u,v,z,intrinsics):
    """Live-test record for one detection at pixel (u,v) with depth z in metres (None if unknown)."""
    class_id=int(boxes.cls[index].item()) if boxes is not None and index<len(boxes) else 0; score=float(boxes.conf[index].item()) if boxes is not None and index<len(boxes) else 0.0
    xyz=deproject_pixel(u,v,z,intrinsics)
    class_name=names.get(class_id,class_id) if isinstance(names,dict) else names[class_id]
    return {"class_id":class_id,"class_name":str(class_name),"confidence":round(score,4),"distance_m":round(z,4) if z is not None else None,"xyz_m":[round(x,4) for x in xyz] if xyz else None}

def draw_annotation_preview(image,objects):
    preview=image.copy(); overlay=image.copy()
    for index,obj in enumerate(objects):
        polygon=np.asarray(obj.polygon_xy,dtype=np.int32)
        color=((37*index+20)%255,(97*index+180)%255,(53*index+80)%255)
        cv2.fillPoly(overlay,[polygon],color); cv2.polylines(preview,[polygon],True,color,2)
        x,y=polygon.min(axis=0); cv2.putText(preview,obj.class_name,(int(x),max(15,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    return cv2.addWeighted(preview,.65,overlay,.35,0)

class State:
    def __init__(self, source, root):
        self.source_name=source; self.root=root.resolve(); self.lock=threading.RLock(); self.stop=threading.Event()
        self.connected=False; self.error=""; self.fps=0.; self.config={"min_depth":.1,"max_depth":1.5}
        self.images={}; self.job=None; self.last_capture=None; self.writer=None; self.last_saved_signature=None
        self.latest_frame=None; self.latest_sequence=0
        self.preprocess_job=None
        self.annotation_job=None
        self.training_job=None; self.preprocess_process=None; self.annotation_process=None; self.training_process=None
        self.test_job=None; self.test_thread=None; self.test_stop=threading.Event(); self.test_image=None
        self.failure_session=None; self.failure_last_signature=None
    def start(self): threading.Thread(target=self.run,daemon=True).start()
    def run(self):
        try:
            with create_frame_source(self.source_name) as source:
                self.connected=True; tick=time.monotonic(); count=0
                for frame in source.frames():
                    if self.stop.is_set(): break
                    with self.lock:
                        cfg=DepthFilterConfig(self.config["min_depth"],self.config["max_depth"])
                        filtered,mask=apply_depth_filter(frame.color_bgr,frame.depth_m,cfg,frame.intrinsics,dim_background=True)
                        self.images={"raw":jpeg(frame.color_bgr),"filtered":jpeg(filtered),"mask":jpeg(cv2.cvtColor(mask.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR))}
                        # Keep one current RGB-D frame ready for Live Test and point-cloud export.
                        # This avoids a black preview while the test thread waits for a post-start frame.
                        self.latest_frame=(frame.color_bgr.copy(),frame.depth_m.copy(),frame.intrinsics,frame.index,frame.timestamp,frame.source); self.latest_sequence+=1
                        if self.job:
                            self.job["seen"]+=1
                            if (self.job["seen"]-1)%self.job["every"]==0:
                                signature=cv2.resize(cv2.cvtColor(frame.color_bgr,cv2.COLOR_BGR2GRAY),(32,24)).astype(np.float32)
                                difference=float(np.mean(np.abs(signature-self.last_saved_signature))) if self.last_saved_signature is not None else 999.0
                                if difference < self.job["similarity_threshold"]:
                                    self.job["skipped_similar"]+=1
                                    continue
                                self.writer.write(frame); self.last_saved_signature=signature; self.job["saved"]+=1
                                if self.job["saved"]>=self.job["target"]: self.finish()
                    count+=1; now=time.monotonic()
                    if now-tick>=1: self.fps=count/(now-tick); tick=now; count=0
                    # The synthetic renderer has no hardware frame clock. Limit it so
                    # it cannot consume every CPU core and starve the inference thread.
                    if self.source_name=="sim": self.stop.wait(1/30)
        except Exception as e: self.error=str(e)
        finally: self.connected=False; self.finish()
    def begin(self,name,target,every,notes):
        if not SAFE.fullmatch(name): raise ValueError("Name may only contain letters, numbers, dots, dashes, and underscores")
        if not 1<=target<=100000 or not 1<=every<=1000: raise ValueError("Frame count or sampling interval is invalid")
        if len(notes)>2000: raise ValueError("Notes are limited to 2000 characters")
        with self.lock:
            if self.job: raise ValueError("A capture is already in progress")
            self.writer=CaptureSessionWriter(self.root/name,purpose="object-segmentation",notes=notes)
            self.job={"name":name,"target":target,"every":every,"saved":0,"seen":0,
                      "skipped_similar":0,"similarity_threshold":2.0}
            self.last_capture=None
            self.last_saved_signature=None
    def finish(self):
        completed_name=None
        with self.lock:
            if self.job:
                completed_name=self.job["name"]
                self.last_capture=dict(self.job)
                self.last_capture["state"]="completed" if self.job["saved"]>=self.job["target"] else "stopped"
                self.last_capture["progress"]=self.job["saved"]/self.job["target"]
            if self.writer: self.writer.close()
            owner=sudo_owner()
            if self.job and owner:
                uid,gid=owner; root=self.root/self.job["name"]
                for path in [root,*root.rglob("*")]:
                    try: os.chown(path,uid,gid)
                    except OSError: pass
            self.writer=None; self.job=None
        if completed_name:
            subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(self.root.parents[1])],capture_output=True)
    def status(self):
        with self.lock:
            job=dict(self.job) if self.job else None
            if job: job["progress"]=job["saved"]/job["target"]
            return {"connected":self.connected,"error":self.error,"fps":round(self.fps,1),"config":self.config,"capture":job,"last_capture":dict(self.last_capture) if self.last_capture else None}
    def sessions(self):
        out=[]
        if not self.root.exists(): return out
        for p in self.root.iterdir():
            try: meta=json.loads((p/"session.json").read_text())
            except Exception: continue
            imgs=sorted(x.name for x in (p/"color").glob("*.png"))
            out.append({"name":p.name,"frame_count":meta.get("frame_count",len(imgs)),"created_utc":meta.get("created_utc",""),"notes":meta.get("notes",""),"thumbnails":imgs[:4]})
        return sorted(out,key=lambda x:x["created_utc"],reverse=True)
    def session_detail(self,name):
        if not SAFE.fullmatch(name): raise FileNotFoundError
        session=(self.root/name).resolve()
        if self.root not in session.parents or not (session/"session.json").is_file(): raise FileNotFoundError
        meta=json.loads((session/"session.json").read_text(encoding="utf-8"))
        frames=sorted(x.name for x in (session/"color").glob("frame_*.png"))
        return {"name":name,"metadata":meta,"frames":frames}
    def delete_session(self,name,confirmation):
        if not SAFE.fullmatch(name) or confirmation!=name:
            raise ValueError("Type the exact session name to confirm deletion")
        with self.lock:
            if self.job and self.job["name"]==name: raise ValueError("Cannot delete a session while it is being captured")
            if self.preprocess_job and self.preprocess_job.get("state")=="running" and self.preprocess_job.get("session")==name: raise ValueError("Cannot delete this session while preprocessing is running")
            if self.annotation_job and self.annotation_job.get("state")=="running" and self.annotation_job.get("session")==name: raise ValueError("Cannot delete this session while annotation is running")
        used_by=self._exports_using_session(name)
        if used_by: raise ValueError(f"Delete dependent dataset export(s) first: {', '.join(used_by)}")
        raw=(self.root/name).resolve(); data=self.root.parents[1]
        if self.root not in raw.parents or not (raw/"session.json").is_file(): raise FileNotFoundError(name)
        targets=[raw,data/"derived"/"object"/name,data/"annotations"/"object_segmentation"/name]
        for target in targets:
            if target.is_symlink(): target.unlink()
            elif target.exists(): shutil.rmtree(target)
        for link in (ROOT/"datasets"/"captures"/name,
                     ROOT/"datasets"/"object_segmentation"/name):
            if link.is_symlink(): link.unlink()
        catalog=data/"catalog"/"index.json"
        if catalog.exists():
            current=json.loads(catalog.read_text(encoding="utf-8")); current["sessions"]=[s for s in current.get("sessions",[]) if s.get("id")!=name]
            current["generated_utc"]=datetime.now().astimezone().isoformat(); catalog.write_text(json.dumps(current,indent=2)+"\n",encoding="utf-8")
        return {"name":name,"deleted":[str(x) for x in targets]}

    def _exports_using_session(self,name):
        exports=self.root.parents[1]/"exports"/"object_segmentation"; used=[]
        if not exports.exists(): return used
        for manifest in exports.glob("*/manifest.json"):
            try:
                data=json.loads(manifest.read_text(encoding="utf-8"))
                if any(item.get("session")==name for item in data.get("items",[])): used.append(manifest.parent.name)
            except (OSError,json.JSONDecodeError): pass
        return sorted(used)

    def storage_report(self):
        data=self.root.parents[1]; categories={"raw":data/"raw","derived":data/"derived","annotations":data/"annotations","exports":data/"exports","models":data/"models","runs":data/"runs"}; result={}; global_inodes=set(); total_unique=0
        for name,root in categories.items():
            apparent=0; unique=0; files=0; inodes=set()
            if root.exists():
                for base,_,names in os.walk(root,followlinks=False):
                    for filename in names:
                        path=Path(base)/filename
                        try:
                            stat=path.stat(follow_symlinks=False)
                            if not path.is_file() or path.is_symlink(): continue
                        except OSError: continue
                        key=(stat.st_dev,stat.st_ino); apparent+=stat.st_size; files+=1
                        if key not in inodes: unique+=stat.st_size; inodes.add(key)
                        if key not in global_inodes: total_unique+=stat.st_size; global_inodes.add(key)
            result[name]={"bytes":apparent,"unique_bytes":unique,"files":files}
        return {"categories":result,"physical_bytes":total_unique,"note":"Hard-linked export images share storage with raw RGB files and are counted once in physical_bytes."}
    def image(self,s,f):
        if not SAFE.fullmatch(s) or not re.fullmatch(r"frame_\d{6}\.png",f): raise FileNotFoundError
        p=(self.root/s/"color"/f).resolve()
        if self.root not in p.parents or not p.is_file(): raise FileNotFoundError
        return p
    def preprocess_start(self,data):
        name=str(data.get("session","")); version=str(data.get("version",""))
        if not SAFE.fullmatch(name) or not SAFE.fullmatch(version): raise ValueError("Invalid session or version name")
        source=(self.root/name).resolve(); data_root=self.root.parents[1]; output=data_root/"derived"/"object"/name/version
        if self.root not in source.parents or not (source/"session.json").is_file(): raise ValueError("Session not found")
        if output.exists() and any(output.iterdir()): raise ValueError("This derived version already exists; choose a new version name")
        minimum=float(data.get("min_depth",.1)); maximum=float(data.get("max_depth",1.5)); height=float(data.get("min_table_height",.015))
        use_roi=bool(data.get("use_roi",True)); roi=str(data.get("roi","0.15,0.25,0.85,0.80"))
        if not .001<=minimum<maximum<=10 or not 0<=height<=1: raise ValueError("Invalid depth or table-height settings")
        with self.lock:
            if self.preprocess_job and self.preprocess_job["state"]=="running": raise ValueError("Preprocessing is already running")
            self.preprocess_job={"state":"running","session":name,"version":version,"error":"","started":time.time()}
        command=[sys.executable,str(ROOT/"tools"/"preprocess_rgbd_session.py"),str(source),"--out",str(output),"--min-depth",str(minimum),"--max-depth",str(maximum),"--min-table-height",str(height)]
        if use_roi: command.extend(["--roi",roi])
        if bool(data.get("auto_table",True)): command.append("--auto-table")
        threading.Thread(target=self._run_preprocess,args=(command,output),daemon=True).start()
        return dict(self.preprocess_job)
    def _run_preprocess(self,command,output):
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        with self.lock: self.preprocess_process=process
        stdout,stderr=process.communicate(); returncode=process.returncode
        with self.lock:
            self.preprocess_job["state"]="complete" if returncode==0 else "failed"
            self.preprocess_job["error"]=(stderr or stdout).strip() if returncode else ""
            self.preprocess_job["details"]=stdout.strip()
            self.preprocess_job["finished"]=time.time(); self.preprocess_process=None
        if returncode==0:
            owner=sudo_owner()
            if owner:
                uid,gid=owner
                for path in [output,*output.rglob("*")]:
                    try: os.chown(path,uid,gid)
                    except OSError: pass
            subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(self.root.parents[1])],capture_output=True)
        elif output.exists():
            shutil.rmtree(output)
    def preprocess_status(self):
        with self.lock: return dict(self.preprocess_job) if self.preprocess_job else {"state":"idle"}
    def derived_result(self,name,version):
        if not SAFE.fullmatch(name) or not SAFE.fullmatch(version): raise FileNotFoundError
        root=self.root.parents[1]/"derived"/"object"/name/version
        if not (root/"config.json").is_file(): raise FileNotFoundError
        frames=sorted(x.name for x in (root/"previews").glob("frame_*.png"))
        return {"session":name,"version":version,"config":json.loads((root/"config.json").read_text()),"frames":frames}
    def derived_image(self,name,version,kind,frame):
        if not SAFE.fullmatch(name) or not SAFE.fullmatch(version) or kind not in {"previews","masks"} or not re.fullmatch(r"frame_\d{6}\.png",frame): raise FileNotFoundError
        root=(self.root.parents[1]/"derived"/"object"/name/version).resolve(); path=(root/kind/frame).resolve()
        if root not in path.parents or not path.is_file(): raise FileNotFoundError
        return path
    def preprocess_delete(self,name,version,confirmation):
        if not SAFE.fullmatch(name) or not SAFE.fullmatch(version) or confirmation!=version:
            raise ValueError("Type the exact derived version name to confirm deletion")
        data=self.root.parents[1]; target=(data/"derived"/"object"/name/version).resolve(); parent=(data/"derived"/"object"/name).resolve()
        if parent not in target.parents or not (target/"config.json").is_file(): raise ValueError("Derived version not found")
        with self.lock:
            if self.preprocess_job and self.preprocess_job.get("state")=="running" and self.preprocess_job.get("session")==name and self.preprocess_job.get("version")==version:
                raise ValueError("Cannot delete a derived version while it is being generated")
            if self.annotation_job and self.annotation_job.get("state")=="running" and self.annotation_job.get("session")==name and self.annotation_job.get("version")==version:
                raise ValueError("Cannot delete a derived version while annotation is using it")
        annotation_run=data/"annotations"/"object_segmentation"/name/"annotation_run.json"
        if annotation_run.is_file():
            try:
                source_value=json.loads(annotation_run.read_text(encoding="utf-8")).get("derived_masks")
                if source_value and Path(source_value).resolve()==target: raise ValueError("This version is used by saved annotations and cannot be deleted")
            except json.JSONDecodeError: pass
        shutil.rmtree(target)
        subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(data)],capture_output=True)
        return {"session":name,"version":version,"deleted":str(target)}
    def annotation_start(self,data):
        name=str(data.get("session","")); version=str(data.get("version","")); method=str(data.get("method","depth"))
        if not SAFE.fullmatch(name) or not SAFE.fullmatch(version): raise ValueError("Invalid session or preprocessing version")
        if method not in {"depth","mobile_sam"}: raise ValueError("Unknown annotation method")
        source=(self.root/name).resolve(); data_root=self.root.parents[1]
        raw_bypass=version=="__raw__"
        derived=(data_root/"derived"/"object"/name/version).resolve() if not raw_bypass else None
        output=(data_root/"annotations"/"object_segmentation"/name).resolve()
        if self.root not in source.parents or not (source/"session.json").is_file(): raise ValueError("Session not found")
        if not raw_bypass and not (derived/"manifest.jsonl").is_file(): raise ValueError("Run preprocessing for this session first, or choose Raw RGB-D")
        class_id=int(data.get("class_id",0)); class_name=str(data.get("class_name","object")).strip()
        min_area=int(data.get("min_area",500)); max_instances=int(data.get("max_instances",20))
        if class_id<0 or not SAFE.fullmatch(class_name): raise ValueError("Invalid class id or class name")
        if not 1<=min_area<=10000000 or not 1<=max_instances<=100: raise ValueError("Invalid proposal settings")
        command=[sys.executable,str(ROOT/"tools"/"annotate_object_segmentation.py"),str(source),"--out",str(output),"--class-id",str(class_id),"--class-name",class_name,"--min-area",str(min_area),"--max-instances",str(max_instances)]
        command.extend(["--raw"] if raw_bypass else ["--derived",str(derived)])
        progress_file=output/".annotation_progress.json"
        try: progress_file.unlink()
        except FileNotFoundError: pass
        command.extend(["--progress-file",str(progress_file)])
        model=data_root/"models"/"annotation"/"mobile_sam.pt"
        if method=="mobile_sam":
            if not model.is_file(): raise ValueError(f"MobileSAM model not found: {model}")
            command.extend(["--sam-model",str(model)])
        if bool(data.get("force",False)): command.append("--force")
        total=len(list((source/"color").glob("frame_*.png")))
        with self.lock:
            if self.annotation_job and self.annotation_job["state"]=="running": raise ValueError("Annotation is already running")
            self.annotation_job={"state":"running","session":name,"version":"raw_rgbd" if raw_bypass else version,"method":method,"output":str(output),"progress_file":str(progress_file),"completed":0,"generated":0,"preserved":0,"total":total,"error":"","started":time.time()}
        threading.Thread(target=self._run_annotation,args=(command,output),daemon=True).start()
        return self.annotation_status()
    def _run_annotation(self,command,output):
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        with self.lock: self.annotation_process=process
        stdout,stderr=process.communicate(); returncode=process.returncode
        progress={}
        progress_file=output/".annotation_progress.json"
        try: progress=json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): pass
        try: progress_file.unlink()
        except FileNotFoundError: pass
        with self.lock:
            self.annotation_job["state"]="complete" if returncode==0 else "failed"
            self.annotation_job["error"]=(stderr or stdout).strip() if returncode else ""
            self.annotation_job["completed"]=int(progress.get("processed",self.annotation_job.get("completed",0)))
            self.annotation_job["generated"]=int(progress.get("generated",self.annotation_job.get("generated",0)))
            self.annotation_job["preserved"]=int(progress.get("preserved",self.annotation_job.get("preserved",0)))
            self.annotation_job["finished"]=time.time(); self.annotation_process=None
        if returncode==0:
            owner=sudo_owner()
            if owner:
                uid,gid=owner
                for path in [output,*output.rglob("*")]:
                    try: os.chown(path,uid,gid)
                    except OSError: pass
            subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(self.root.parents[1])],capture_output=True)
    def annotation_status(self):
        with self.lock:
            if not self.annotation_job: return {"state":"idle"}
            job=dict(self.annotation_job)
        progress_file=Path(job.get("progress_file",""))
        if progress_file.is_file():
            try:
                progress=json.loads(progress_file.read_text(encoding="utf-8"))
                job["completed"]=int(progress.get("processed",0)); job["generated"]=int(progress.get("generated",0)); job["preserved"]=int(progress.get("preserved",0))
                with self.lock:
                    if self.annotation_job:
                        self.annotation_job.update({key:job[key] for key in ("completed","generated","preserved")})
            except (OSError,json.JSONDecodeError,TypeError,ValueError): pass
        job["progress"]=job["completed"]/job["total"] if job["total"] else 0
        return job
    def annotation_result(self,name):
        if not SAFE.fullmatch(name): raise FileNotFoundError
        root=self.root.parents[1]/"annotations"/"object_segmentation"/name
        run=root/"annotation_run.json"
        if not run.is_file(): raise FileNotFoundError
        frames=sorted(x.name for x in (root/"previews").glob("frame_*.png"))
        states={"approved":0,"rejected":0,"pending":0}; objects=0
        for path in (root/"annotations").glob("frame_*.json"):
            try:
                item=json.loads(path.read_text(encoding="utf-8")); state=item.get("review_status","pending")
                states[state]=states.get(state,0)+1; objects+=len(item.get("objects",[]))
            except (OSError,json.JSONDecodeError): pass
        run_data=json.loads(run.read_text(encoding="utf-8"))
        run_data.setdefault("preprocessing_version",Path(run_data.get("derived_masks","")).name or "unknown")
        method=run_data.get("proposal_method","depth-components")
        run_data["method_label"]=("MobileSAM automatic masks on Raw RGB" if method.startswith("sam-auto:") else "Manual annotation on Raw RGB" if method=="manual-raw" else "MobileSAM + depth proposals" if method.startswith("sam-box:") else "Depth components only")
        return {"session":name,"run":run_data,"frames":frames,"states":states,"objects":objects}

    def annotation_delete(self,name,confirmation):
        if not SAFE.fullmatch(name) or confirmation!=name: raise ValueError("Type the exact annotation session name to confirm deletion")
        used_by=self._exports_using_session(name)
        if used_by: raise ValueError(f"Delete dependent dataset export(s) first: {', '.join(used_by)}")
        root=(self.root.parents[1]/"annotations"/"object_segmentation").resolve(); target=(root/name).resolve()
        if root not in target.parents or not (target/"annotation_run.json").is_file(): raise ValueError("Annotation set not found")
        with self.lock:
            if self.annotation_job and self.annotation_job.get("state")=="running" and self.annotation_job.get("session")==name: raise ValueError("Cannot delete annotations while generation is running")
        shutil.rmtree(target)
        subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(self.root.parents[1])],capture_output=True)
        return {"session":name,"deleted":str(target),"preserved":["raw","derived"]}
    def annotation_image(self,name,frame):
        if not SAFE.fullmatch(name) or not re.fullmatch(r"frame_\d{6}\.png",frame): raise FileNotFoundError
        root=(self.root.parents[1]/"annotations"/"object_segmentation"/name/"previews").resolve(); path=(root/frame).resolve()
        if root not in path.parents or not path.is_file(): raise FileNotFoundError
        return path
    def review_session(self,name):
        result=self.annotation_result(name)
        root=self.root.parents[1]/"annotations"/"object_segmentation"/name/"annotations"
        states={}
        for path in root.glob("frame_*.json"):
            try: states[f"{path.stem}.png"]=json.loads(path.read_text(encoding="utf-8")).get("review_status","pending")
            except (OSError,json.JSONDecodeError): pass
        result["frames"]=sorted(states); result["frame_states"]=states
        return result
    def review_frame(self,name,frame):
        if not SAFE.fullmatch(name) or not re.fullmatch(r"frame_\d{6}\.png",frame): raise FileNotFoundError
        path=self.root.parents[1]/"annotations"/"object_segmentation"/name/"annotations"/f"{Path(frame).stem}.json"
        if not path.is_file(): raise FileNotFoundError
        return json.loads(path.read_text(encoding="utf-8"))
    def review_save(self,data):
        name=str(data.get("session","")); frame=str(data.get("frame","")); state=str(data.get("review_status","pending"))
        if not SAFE.fullmatch(name) or not re.fullmatch(r"frame_\d{6}\.png",frame): raise ValueError("Invalid session or frame")
        if state not in {"pending","approved","rejected"}: raise ValueError("Invalid review status")
        data_root=self.root.parents[1]; annotation_root=(data_root/"annotations"/"object_segmentation"/name).resolve()
        path=(annotation_root/"annotations"/f"{Path(frame).stem}.json").resolve()
        if annotation_root not in path.parents or not path.is_file(): raise ValueError("Annotation frame not found")
        existing=json.loads(path.read_text(encoding="utf-8")); width=int(existing["width"]); height=int(existing["height"])
        raw_objects=data.get("objects",[])
        if not isinstance(raw_objects,list) or len(raw_objects)>100: raise ValueError("Invalid object list")
        objects=[]
        for index,item in enumerate(raw_objects):
            if not isinstance(item,dict): raise ValueError(f"Object {index+1} is invalid")
            class_id=int(item.get("class_id",0)); class_name=str(item.get("class_name","")).strip()
            if class_id<0 or not class_name or len(class_name)>80 or any(ord(c)<32 for c in class_name): raise ValueError(f"Object {index+1} has an invalid class")
            raw_polygon=item.get("polygon_xy",[])
            if not isinstance(raw_polygon,list) or not 3<=len(raw_polygon)<=5000: raise ValueError(f"Object {index+1} needs at least 3 polygon points")
            polygon=[]
            for point in raw_polygon:
                if not isinstance(point,(list,tuple)) or len(point)!=2: raise ValueError(f"Object {index+1} has an invalid point")
                x=float(point[0]); y=float(point[1])
                if not np.isfinite([x,y]).all() or not 0<=x<width or not 0<=y<height: raise ValueError(f"Object {index+1} has a point outside the image")
                polygon.append([x,y])
            points=np.asarray(polygon); bbox=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
            objects.append(ObjectAnnotation(class_id,class_name,polygon,bbox,str(item.get("source") or "manual"),item.get("confidence")))
        annotation=FrameAnnotation(frame_id=existing["frame_id"],source_session=existing["source_session"],source_color=existing["source_color"],width=width,height=height,objects=objects,review_status=state,schema=existing.get("schema","armsoft.object-segmentation.v1"))
        temporary=path.with_suffix(".json.tmp"); annotation.save(temporary); os.replace(temporary,path)
        image=cv2.imread(str(self.image(name,frame)),cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Raw image cannot be read")
        preview=annotation_root/"previews"/frame; preview.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(preview),draw_annotation_preview(image,objects),[cv2.IMWRITE_PNG_COMPRESSION,9]): raise ValueError("Preview could not be saved")
        owner=sudo_owner()
        if owner:
            uid,gid=owner
            for item in (path,preview):
                try: os.chown(item,uid,gid)
                except OSError: pass
        subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(data_root)],capture_output=True)
        return {"session":name,"frame":frame,"review_status":state,"objects":len(objects)}
    def dataset_inputs(self):
        data_root=self.root.parents[1]; root=data_root/"annotations"/"object_segmentation"; sessions=[]
        if not root.exists(): return {"sessions":[]}
        for project in sorted(x for x in root.iterdir() if x.is_dir()):
            states={"approved":0,"rejected":0,"pending":0}; objects=0; classes={}
            for path in (project/"annotations").glob("frame_*.json"):
                try:
                    item=json.loads(path.read_text(encoding="utf-8")); state=item.get("review_status","pending"); states[state]=states.get(state,0)+1
                    for obj in item.get("objects",[]): classes[str(obj["class_id"])]=obj["class_name"]; objects+=1
                except (OSError,json.JSONDecodeError,KeyError,TypeError): pass
            if sum(states.values()): sessions.append({"id":project.name,"states":states,"objects":objects,"classes":classes})
        return {"sessions":sessions}
    def _dataset_plan(self,data):
        assignments=data.get("assignments",[]); errors=[]; warnings=[]; selected=[]; seen=set(); source_splits={}; classes={}; splits={x:{"sessions":[],"images":0,"objects":0} for x in ("train","val","test")}
        if not isinstance(assignments,list) or len(assignments)>1000: return {"valid":False,"errors":["Assignments must be a list with at most 1000 sessions"],"warnings":[],"splits":splits,"classes":{}}
        for item in assignments:
            name=str(item.get("session","")); split=str(item.get("split",""))
            if not SAFE.fullmatch(name) or split not in splits: errors.append(f"Invalid assignment: {name or 'unnamed'} → {split or 'none'}"); continue
            if name in seen: errors.append(f"Session {name} is assigned more than once"); continue
            seen.add(name); project=self.root.parents[1]/"annotations"/"object_segmentation"/name; annotation_dir=project/"annotations"
            if not (project/"annotation_run.json").is_file(): errors.append(f"Annotation session not found: {name}"); continue
            session_record={"session":name,"split":split,"frames":[]}; splits[split]["sessions"].append(name)
            for path in sorted(annotation_dir.glob("frame_*.json")):
                try: annotation=FrameAnnotation.load(path)
                except Exception as exc: errors.append(f"{name}/{path.name}: {exc}"); continue
                if annotation.review_status!="approved": continue
                expected_source=(self.root/name).resolve(); actual_source=Path(annotation.source_session).resolve()
                if actual_source!=expected_source: errors.append(f"{name}/{annotation.frame_id}: source session does not match its annotation set"); continue
                source_key=str(actual_source); previous_split=source_splits.get(source_key)
                if previous_split is not None and previous_split!=split: errors.append(f"Source session {Path(source_key).name} appears in both {previous_split} and {split}")
                source_splits[source_key]=split
                source=Path(annotation.source_session)/annotation.source_color
                if not source.is_file(): errors.append(f"Missing RGB image for {name}/{annotation.frame_id}"); continue
                for obj in annotation.objects:
                    previous=classes.get(obj.class_id)
                    if previous is not None and previous!=obj.class_name: errors.append(f"Class ID {obj.class_id} conflicts: {previous!r} vs {obj.class_name!r}")
                    classes[obj.class_id]=obj.class_name
                session_record["frames"].append((annotation,path,source)); splits[split]["images"]+=1; splits[split]["objects"]+=len(annotation.objects)
            if not session_record["frames"]: warnings.append(f"{name} has no approved frames and will add nothing")
            selected.append(session_record)
        if not selected: errors.append("Select at least one annotation session")
        if splits["train"]["images"]==0: errors.append("The train split needs at least one approved frame")
        if splits["val"]["images"]==0: errors.append("The validation split needs at least one approved frame")
        elif splits["val"]["images"]<10: warnings.append("The validation split has fewer than 10 images; metrics will be unstable")
        if splits["train"]["images"]+splits["val"]["images"]<50: warnings.append("This dataset has fewer than 50 train/validation images; use it to test the pipeline, not model quality")
        if splits["test"]["images"]==0: warnings.append("The test split is empty; this is acceptable for an early dataset")
        return {"valid":not errors,"errors":errors,"warnings":warnings,"splits":splits,"classes":{str(k):v for k,v in sorted(classes.items())},"_selected":selected}
    def dataset_validate(self,data):
        plan=self._dataset_plan(data); plan.pop("_selected",None); return plan
    def dataset_export(self,data):
        version=str(data.get("version",""))
        if not SAFE.fullmatch(version): raise ValueError("Dataset version may only contain letters, numbers, dots, dashes, and underscores")
        plan=self._dataset_plan(data)
        if not plan["valid"]: raise ValueError("; ".join(plan["errors"]))
        data_root=self.root.parents[1]; exports=data_root/"exports"/"object_segmentation"; exports.mkdir(parents=True,exist_ok=True); destination=exports/version
        if destination.exists(): raise ValueError("This dataset version already exists; choose a new version name")
        stage=Path(tempfile.mkdtemp(prefix=f".{version}-",dir=exports)); manifest=[]; linked=0; copied=0
        try:
            for split in ("train","val","test"):
                (stage/"images"/split).mkdir(parents=True); (stage/"labels"/split).mkdir(parents=True)
            for session in plan["_selected"]:
                split=session["split"]
                for annotation,annotation_path,source in session["frames"]:
                    image_name=f"{session['session']}__{annotation.frame_id}{source.suffix.lower() or '.png'}"; target=stage/"images"/split/image_name
                    try: os.link(source,target); linked+=1
                    except OSError: shutil.copy2(source,target); copied+=1
                    label=stage/"labels"/split/f"{Path(image_name).stem}.txt"; lines=to_yolo_lines(annotation); label.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
                    manifest.append({"session":session["session"],"split":split,"image":str(target.relative_to(stage)),"label":str(label.relative_to(stage)),"annotation":str(annotation_path.resolve()),"objects":len(annotation.objects)})
            names="\n".join(f"  {key}: {json.dumps(value)}" for key,value in sorted(((int(k),v) for k,v in plan["classes"].items())))
            (stage/"data.yaml").write_text(f"path: {destination.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",encoding="utf-8")
            public={k:v for k,v in plan.items() if k!="_selected"}; metadata={"schema":"armsoft.yolo-seg-export.v1","version":version,"created_utc":datetime.now().astimezone().isoformat(),"approved_only":True,"storage":{"hardlinked_images":linked,"copied_images":copied},"plan":public,"items":manifest}
            (stage/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8"); os.replace(stage,destination)
        except Exception:
            shutil.rmtree(stage,ignore_errors=True); raise
        owner=sudo_owner()
        if owner:
            uid,gid=owner
            for path in [destination,*destination.rglob("*")]:
                try: os.chown(path,uid,gid)
                except OSError: pass
        public={k:v for k,v in plan.items() if k!="_selected"}; return {"version":version,"path":str(destination),"storage":{"hardlinked_images":linked,"copied_images":copied},"plan":public}
    def dataset_exports(self):
        root=self.root.parents[1]/"exports"/"object_segmentation"; exports=[]
        if not root.exists(): return {"exports":[]}
        for path in sorted((x for x in root.iterdir() if x.is_dir() and not x.name.startswith('.')),reverse=True):
            manifest=path/"manifest.json"
            if not manifest.is_file(): continue
            try:
                data=json.loads(manifest.read_text(encoding="utf-8")); exports.append({"version":path.name,"created_utc":data.get("created_utc"),"splits":data.get("plan",{}).get("splits",{}),"classes":data.get("plan",{}).get("classes",{}),"storage":data.get("storage",{}),"path":str(path)})
            except (OSError,json.JSONDecodeError): pass
        return {"exports":exports}
    def dataset_delete(self,version,confirmation):
        if not SAFE.fullmatch(version) or confirmation!=version: raise ValueError("Type the exact dataset version to confirm deletion")
        root=(self.root.parents[1]/"exports"/"object_segmentation").resolve(); target=(root/version).resolve()
        if root not in target.parents or not (target/"manifest.json").is_file(): raise ValueError("Dataset export not found")
        with self.lock:
            if self.training_job and self.training_job.get("state") in {"running","stopping"} and self.training_job.get("dataset")==version: raise ValueError("Stop the training run before deleting its dataset")
        shutil.rmtree(target); return {"version":version,"deleted":str(target)}

    def training_inputs(self):
        exports=self.dataset_exports()["exports"]
        devices=compute_devices(recommended=True)
        return {"datasets":exports,"models":[
            {"id":"yolo11n-seg.pt","label":"YOLO11n Seg · fastest"},
            {"id":"yolo11s-seg.pt","label":"YOLO11s Seg · more accurate"},
            {"id":"yolo11m-seg.pt","label":"YOLO11m Seg · heavier"}],"devices":devices}

    def _training_paths(self,dataset,run_name):
        data=self.root.parents[1]; dataset_root=(data/"exports"/"object_segmentation"/dataset).resolve(); runs_root=(data/"runs"/"object_segmentation").resolve()
        return dataset_root,runs_root,(runs_root/run_name).resolve()

    def training_start(self,data):
        dataset=str(data.get("dataset","")); run_name=str(data.get("run_name","")); model=str(data.get("model","")); device=str(data.get("device","cpu"))
        if not SAFE.fullmatch(dataset) or not SAFE.fullmatch(run_name): raise ValueError("Dataset and run names may only contain letters, numbers, dots, dashes, and underscores")
        if model not in {"yolo11n-seg.pt","yolo11s-seg.pt","yolo11m-seg.pt"}: raise ValueError("Choose one of the supported YOLO segmentation checkpoints")
        if device not in {item["id"] for item in compute_devices()}: raise ValueError("Choose an available compute device")
        try: epochs=int(data.get("epochs",100)); imgsz=int(data.get("imgsz",640)); batch=int(data.get("batch",8))
        except (TypeError,ValueError): raise ValueError("Epochs, image size, and batch must be whole numbers")
        if not 1<=epochs<=10000: raise ValueError("Epochs must be between 1 and 10000")
        if not 128<=imgsz<=2048 or imgsz%32: raise ValueError("Image size must be a multiple of 32 between 128 and 2048")
        if not 1<=batch<=512: raise ValueError("Batch size must be between 1 and 512")
        dataset_root,runs_root,run_dir=self._training_paths(dataset,run_name)
        if not (dataset_root/"data.yaml").is_file() or not (dataset_root/"manifest.json").is_file(): raise ValueError("Dataset export is incomplete or does not exist")
        if runs_root not in run_dir.parents: raise ValueError("Invalid run path")
        with self.lock:
            if self.training_job and self.training_job.get("state")=="running": raise ValueError("A training run is already in progress")
            if self.test_job and self.test_job.get("state") in {"loading","running"}: raise ValueError("Stop live testing before starting training")
            if run_dir.exists(): raise ValueError("This run name already exists; choose a new one")
            runs_root.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,"-u",str(ROOT/"tools"/"train_object_segmentation.py"),"--data",str(dataset_root/"data.yaml"),"--model",model,"--epochs",str(epochs),"--imgsz",str(imgsz),"--batch",str(batch),"--device",device,"--project",str(runs_root),"--name",run_name]
            self.training_job={"state":"running","dataset":dataset,"run_name":run_name,"model":model,"device":device,"epochs":epochs,"imgsz":imgsz,"batch":batch,"completed_epochs":0,"progress":0.0,"started_utc":datetime.now().astimezone().isoformat(),"finished_utc":None,"log":[],"error":""}
        threading.Thread(target=self._run_training,args=(command,run_dir),daemon=True).start()
        return self.training_status()

    def _run_training(self,command,run_dir):
        try:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT,start_new_session=(os.name=="posix"))
            with self.lock: self.training_process=process; self.training_job["pid"]=process.pid
            assert process.stdout is not None
            for line in process.stdout:
                with self.lock: self.training_job["log"]=(self.training_job["log"]+[line.rstrip()])[-200:]
            code=process.wait()
            with self.lock:
                if self.training_job.get("state")=="stopping": self.training_job["state"]="cancelled"
                elif code==0: self.training_job["state"]="complete"; self.training_job["progress"]=1.0; self.training_job["completed_epochs"]=self.training_job["epochs"]
                else: self.training_job["state"]="failed"; self.training_job["error"]=f"Training exited with code {code}"
                self.training_job["finished_utc"]=datetime.now().astimezone().isoformat(); self.training_process=None
        except Exception as exc:
            with self.lock:
                self.training_job["state"]="failed"; self.training_job["error"]=str(exc); self.training_job["finished_utc"]=datetime.now().astimezone().isoformat(); self.training_process=None
        finally:
            owner=sudo_owner()
            if run_dir.exists() and owner:
                uid,gid=owner
                for path in [run_dir,*run_dir.rglob("*")]:
                    try: os.chown(path,uid,gid)
                    except OSError: pass

    def training_status(self):
        with self.lock: job=dict(self.training_job) if self.training_job else {"state":"idle","log":[],"progress":0.0,"completed_epochs":0}
        if job.get("state") in {"running","stopping"}:
            _,_,run_dir=self._training_paths(job["dataset"],job["run_name"]); results=run_dir/"results.csv"
            if results.is_file():
                try:
                    completed=max(0,sum(1 for _ in results.open(encoding="utf-8"))-1); job["completed_epochs"]=completed; job["progress"]=min(1.0,completed/job["epochs"])
                except OSError: pass
        return job

    def training_stop(self):
        with self.lock:
            if not self.training_job or self.training_job.get("state")!="running" or self.training_process is None: raise ValueError("No training run is active")
            self.training_job["state"]="stopping"
            if os.name=="posix": os.killpg(os.getpgid(self.training_process.pid),signal.SIGTERM)
            else: self.training_process.terminate()
        return self.training_status()

    def shutdown(self):
        self.stop.set(); self.test_stop.set(); self.finish()
        with self.lock: processes=[self.preprocess_process,self.annotation_process,self.training_process]
        for process in processes:
            if process and process.poll() is None:
                try:
                    if process is self.training_process and os.name=="posix": os.killpg(os.getpgid(process.pid),signal.SIGTERM)
                    else: process.terminate()
                except (OSError,ProcessLookupError): pass

    @staticmethod
    def _training_metrics(path):
        results=path/"results.csv"
        if not results.is_file(): return {}
        try:
            with results.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
            if not rows: return {}
            row={k.strip():v.strip() for k,v in rows[-1].items()}; output={}
            for source,target in (("metrics/mAP50(B)","map50"),("metrics/mAP50-95(B)","map50_95"),("metrics/precision(B)","precision"),("metrics/recall(B)","recall")):
                if source in row:
                    try: output[target]=round(float(row[source]),4)
                    except ValueError: pass
            output["epochs_completed"]=len(rows); return output
        except (OSError,csv.Error): return {}

    def training_runs(self):
        root=self.root.parents[1]/"runs"/"object_segmentation"; runs=[]
        if not root.exists(): return {"runs":[]}
        for path in sorted((x for x in root.iterdir() if x.is_dir() and not x.name.startswith('.')),key=lambda x:x.stat().st_mtime,reverse=True):
            metadata={}
            try: metadata=json.loads((path/"armsoft_training.json").read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): pass
            runs.append({"name":path.name,"created_utc":metadata.get("created_utc") or datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),"dataset":metadata.get("dataset",""),"base_model":metadata.get("base_model",""),"best_weights":metadata.get("best_weights") if (path/"weights"/"best.pt").is_file() else None,"metrics":self._training_metrics(path),"complete":bool(metadata),"path":str(path)})
        return {"runs":runs}

    def training_delete(self,name,confirmation):
        if not SAFE.fullmatch(name) or confirmation!=name: raise ValueError("Type the exact run name to confirm deletion")
        root=(self.root.parents[1]/"runs"/"object_segmentation").resolve(); target=(root/name).resolve()
        if root not in target.parents or not target.is_dir(): raise ValueError("Training run not found")
        with self.lock:
            if self.training_job and self.training_job.get("state") in {"running","stopping"} and self.training_job.get("run_name")==name: raise ValueError("Stop this training run before deleting it")
            if self.test_job and self.test_job.get("state") in {"loading","running"} and self.test_job.get("run_name")==name: raise ValueError("Stop live testing before deleting this model")
        shutil.rmtree(target); return {"name":name,"deleted":str(target)}

    def test_inputs(self):
        runs=[]
        bundled_root=ROOT/"models"/"catalog"
        for model in discover_runtime_models(bundled_root):
            runs.append({"id":f"bundled/{model.id}","name":model.display_name,"model":model.base_model,"dataset":model.dataset_version,"task":model.task,"version":model.model_version,"source":"bundled"})
        runtime_root=self.root.parents[1]/"models"/"runtime"
        for model in discover_runtime_models(runtime_root):
            runs.append({"id":f"runtime/{model.id}","name":model.display_name,"model":model.base_model,"dataset":model.dataset_version,"task":model.task,"version":model.model_version,"source":"runtime"})
        for run in self.training_runs()["runs"]:
            if run.get("complete") and run.get("best_weights"):
                base=str(run.get("base_model") or "")
                task="detection" if "-seg" not in base else "segmentation"
                runs.append({"id":f"object/{run['name']}","name":run["name"],"model":base,"dataset":run.get("dataset"),"task":task})
        pose_root=self.root.parents[1]/"runs"/"mechanical_hand_pose"
        if pose_root.exists():
            for path in sorted((p for p in pose_root.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True):
                weights=path/"weights"/"best.pt"; metadata=path/"armsoft_training.json"
                if not weights.is_file() or not metadata.is_file(): continue
                try: details=json.loads(metadata.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError): continue
                runs.append({"id":f"pose/{path.name}","name":path.name,"model":details.get("base_model","YOLO Pose"),"dataset":details.get("dataset",""),"task":"pose"})
        devices=compute_devices()
        return {"runs":runs,"devices":devices,"camera_connected":self.connected}

    def _test_model(self,model_id):
        if not isinstance(model_id,str) or "/" not in model_id: raise ValueError("Select a completed training run with best.pt")
        family,name=model_id.split("/",1)
        if family not in {"object","pose","runtime","bundled"} or not SAFE.fullmatch(name): raise ValueError("Select a valid local model")
        if family in {"runtime","bundled"}:
            model_root=self.root.parents[1]/"models"/"runtime" if family=="runtime" else ROOT/"models"/"catalog"
            records={model.id:model for model in discover_runtime_models(model_root)}
            model=records.get(name)
            if model is None: raise ValueError("Registered model is missing or invalid")
            return {"id":model_id,"name":model.display_name,"task":model.task,"weights":model.weights,"base_model":model.base_model,"version":model.model_version}
        relative=Path("runs")/("object_segmentation" if family=="object" else "mechanical_hand_pose")/name
        root=self.root.parents[1].resolve(); run=(root/relative).resolve(); weights=run/"weights"/"best.pt"; metadata=run/"armsoft_training.json"
        if root not in run.parents or not weights.is_file() or not metadata.is_file(): raise ValueError("Completed best.pt not found for this model")
        details=json.loads(metadata.read_text(encoding="utf-8")); base=str(details.get("base_model") or "")
        task="pose" if family=="pose" else ("segmentation" if "-seg" in base else "detection")
        return {"id":model_id,"name":name,"task":task,"weights":weights,"base_model":base}

    def test_start(self,data):
        model_id=str(data.get("model_id") or (f"object/{data.get('run_name')}" if data.get("run_name") else "")); device=str(data.get("device","cpu"))
        model_record=self._test_model(model_id)
        if device not in {item["id"] for item in compute_devices()}: raise ValueError("Choose an available inference device")
        try: imgsz=int(data.get("imgsz",416)); confidence=float(data.get("confidence",.25)); target_fps=float(data.get("target_fps",10))
        except (TypeError,ValueError): raise ValueError("Invalid live-test settings")
        if not 128<=imgsz<=1280 or imgsz%32: raise ValueError("Image size must be a multiple of 32 between 128 and 1280")
        if not .01<=confidence<=1 or not 1<=target_fps<=30: raise ValueError("Confidence or FPS limit is invalid")
        with self.lock:
            if not self.connected: raise ValueError("Connect the camera before starting live testing")
            if self.training_job and self.training_job.get("state") in {"running","stopping"}: raise ValueError("Wait for training to finish before live testing")
            if self.test_job and self.test_job.get("state") in {"loading","running","stopping"}: raise ValueError("Live testing is already running or stopping")
            self.test_stop.clear(); self.test_image=None
            self.failure_session=None; self.failure_last_signature=None
            self.test_job={"state":"loading","model_id":model_id,"run_name":model_record["name"],"task":model_record["task"],"base_model":model_record["base_model"],"device":device,"imgsz":imgsz,"confidence":confidence,"target_fps":target_fps,"frames":0,"fps":0.0,"latency_ms":0.0,"objects":[],"error":"","failure_saved":0,"failure_skipped_similar":0,"started_utc":datetime.now().astimezone().isoformat()}
            self.test_thread=threading.Thread(target=self._run_live_test,args=(model_record["weights"],),daemon=True); self.test_thread.start()
        return self.test_status()

    def _run_live_test(self,weights):
        try:
            from ultralytics import YOLO
            model=YOLO(str(weights))
            with self.lock:
                if self.test_job: self.test_job["state"]="running"
            last_sequence=-1; fps_ema=0.0; last_camera_frame=time.monotonic()
            while not self.test_stop.is_set():
                with self.lock:
                    item=self.latest_frame; sequence=self.latest_sequence; job=dict(self.test_job or {}); connected=self.connected
                if item is None or sequence==last_sequence:
                    if not connected and time.monotonic()-last_camera_frame>2: raise RuntimeError("Camera disconnected. Reconnect the D405 and restart Live Test.")
                    if time.monotonic()-last_camera_frame>5: raise RuntimeError("Camera is connected but no new RGB-D frames arrived. Stop other camera apps, reconnect the D405, and retry.")
                    time.sleep(.005); continue
                last_sequence=sequence; color,depth,intrinsics,_,_,_=item; started=time.perf_counter()
                last_camera_frame=started
                result=model.predict(source=color,imgsz=job["imgsz"],conf=job["confidence"],device=job["device"],verbose=False)[0]
                latency=(time.perf_counter()-started)*1000; overlay=result.plot()
                objects=[instance.to_dict() for instance in adapt_ultralytics_result(result,depth,intrinsics)]
                instantaneous=1000/latency if latency>0 else 0; fps_ema=instantaneous if not fps_ema else .8*fps_ema+.2*instantaneous
                with self.lock:
                    if not self.test_job: break
                    self.test_image=jpeg(overlay); self.test_job.update({"frames":self.test_job["frames"]+1,"fps":round(fps_ema,1),"latency_ms":round(latency,1),"objects":objects,"last_frame_utc":datetime.now().astimezone().isoformat()})
                delay=max(0,1/job["target_fps"]-(time.perf_counter()-started)); self.test_stop.wait(delay)
        except Exception as exc:
            with self.lock:
                if self.test_job: self.test_job["state"]="failed"; self.test_job["error"]=str(exc)
        finally:
            with self.lock:
                if self.test_job and self.test_job.get("state") not in {"failed"}: self.test_job["state"]="stopped"

    def test_status(self):
        with self.lock: return dict(self.test_job) if self.test_job else {"state":"idle","frames":0,"fps":0.0,"latency_ms":0.0,"objects":[]}

    def test_stop_live(self):
        with self.lock:
            if not self.test_job or self.test_job.get("state") not in {"loading","running"}: raise ValueError("Live testing is not running")
            self.test_stop.set(); self.test_job["state"]="stopping"
        return self.test_status()

    def test_export_pointcloud(self,data):
        with self.lock: item=self.latest_frame; job=dict(self.test_job or {})
        if item is None: raise ValueError("No RGB-D frame is available for point-cloud export")
        try:
            voxel=float(data.get("voxel_size_m",.005)); max_points=int(data.get("max_points",100000))
        except (TypeError,ValueError): raise ValueError("Point-cloud settings are invalid")
        if not 0<=voxel<=.05 or not 100<=max_points<=500000: raise ValueError("Point-cloud settings are outside safe limits")
        color,depth,intrinsics,index,timestamp,source=item
        frame=RGBDFrame(color.copy(),depth.copy(),intrinsics,index,timestamp,source)
        config=PointCloudConfig(min_depth_m=self.config["min_depth"],max_depth_m=self.config["max_depth"],voxel_size_m=voxel,max_points=max_points)
        result=PointCloudPipeline(config).process(frame)
        root=(self.root.parents[1]/"derived"/"pointcloud").resolve(); stamp=datetime.now().strftime("live_%Y%m%d_%H%M%S_%f"); target=(root/stamp).resolve()
        if root not in target.parents or target.exists(): raise ValueError("Could not create a unique point-cloud artifact")
        target.mkdir(parents=True); ply=save_ply(target/"scene.ply",result)
        metadata={"schema":"armsoft.live-pointcloud.v1","created_utc":datetime.now().astimezone().isoformat(),"frame":{"index":index,"timestamp":timestamp,"source":source},"model":{"id":job.get("model_id"),"task":job.get("task"),"objects":job.get("objects",[])},"config":config.to_dict(),**result.summary()}
        (target/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        return {"artifact":stamp,"ply":str(ply),"metadata":str(target/"metadata.json"),"summary":result.summary()}

    def test_save_failure(self,data):
        failure_type=str(data.get("failure_type","other")); note=str(data.get("note","")).strip()
        allowed={"missed_detection","false_positive","bad_mask","wrong_class","other"}
        if failure_type not in allowed: raise ValueError("Choose a valid failure type")
        if len(note)>500: raise ValueError("Failure note is limited to 500 characters")
        with self.lock:
            item=self.latest_frame; job=dict(self.test_job or {})
        if item is None: raise ValueError("No camera frame is available")
        color,depth,intrinsics,index,timestamp,source=item; name=datetime.now().strftime("failure_%Y%m%d_%H%M%S_%f")
        signature=cv2.resize(cv2.cvtColor(color,cv2.COLOR_BGR2GRAY),(32,24)).astype(np.float32)
        with self.lock:
            difference=float(np.mean(np.abs(signature-self.failure_last_signature))) if self.failure_last_signature is not None else 999.0
            if difference<2.0:
                if self.test_job: self.test_job["failure_skipped_similar"]=self.test_job.get("failure_skipped_similar",0)+1
                return {"session":self.failure_session,"saved":self.test_job.get("failure_saved",0) if self.test_job else 0,"skipped_similar":True}
            if self.failure_session is None: self.failure_session=name
            name=self.failure_session; target=self.root/name; color_dir=target/"color"; depth_dir=target/"depth"
            color_dir.mkdir(parents=True,exist_ok=True); depth_dir.mkdir(parents=True,exist_ok=True)
            manifest=target/"manifest.jsonl"; count=sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()) if manifest.exists() else 0
            frame_id=f"frame_{count:06d}"; color_rel=Path("color")/f"{frame_id}.png"; depth_rel=Path("depth")/f"{frame_id}.npy"
            if not cv2.imwrite(str(target/color_rel),color): raise OSError("Could not save failure RGB frame")
            np.save(target/depth_rel,depth.astype(np.float32,copy=False),allow_pickle=False)
            record={"id":frame_id,"color":color_rel.as_posix(),"depth":depth_rel.as_posix(),"source_index":int(index),"timestamp":float(timestamp),"source":source,"valid_depth_fraction":float((depth>0).mean()),"failure_type":failure_type,"failure_note":note,"test_run":job.get("run_name"),"predictions":job.get("objects",[])}
            with manifest.open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,sort_keys=True)+"\n")
            previous={}
            try: previous=json.loads((target/"session.json").read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): pass
            metadata={"schema":SESSION_SCHEMA_VERSION,"created_utc":previous.get("created_utc",datetime.now().astimezone().isoformat()),"purpose":"object-segmentation-failure-cases","notes":f"Live model failure cases from {job.get('run_name','unknown model')}","frame_count":count+1,"intrinsics":intrinsics.to_dict(),"units":{"depth":"metres"},"colour_order_on_disk":"BGR decoded by OpenCV; PNG itself is standard RGB","failure_case_collection":True,"test_run":job.get("run_name")}
            (target/"session.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            self.failure_last_signature=signature
            if self.test_job: self.test_job["failure_saved"]=self.test_job.get("failure_saved",0)+1
            saved=self.test_job.get("failure_saved",count+1) if self.test_job else count+1
        owner=sudo_owner()
        if owner:
            uid,gid=owner
            for path in [target,*target.rglob("*")]:
                try: os.chown(path,uid,gid)
                except OSError: pass
        subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(self.root.parents[1])],capture_output=True)
        return {"session":name,"saved":saved,"skipped_similar":False,"failure_type":failure_type}

Handler=create_studio_handler(WEB,MAX_REQUEST_BYTES)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765); p.add_argument("--source",choices=("camera","sim"),default="camera"); p.add_argument("--data-root",type=Path,default=data_root()); p.add_argument("--capture-root",type=Path); p.add_argument("--allow-remote",action="store_true",help="allow binding beyond localhost; there is no authentication"); a=p.parse_args()
    if a.host not in {"127.0.0.1","localhost","::1"} and not a.allow_remote: p.error("remote binding requires --allow-remote because this local tool has no authentication")
    a.data_root=a.data_root.expanduser().resolve()
    if a.data_root in {Path("/"),Path.home().resolve()}: p.error("--data-root must be a dedicated data directory, not the filesystem or home root")
    capture_root=(a.capture_root.expanduser().resolve() if a.capture_root else a.data_root/"raw"/"object")
    if capture_root.parents[1]!=a.data_root: p.error("--capture-root must be <data-root>/raw/object so all lifecycle operations share one safe boundary")
    capture_root.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/"tools"/"build_data_catalog.py"),"--data-root",str(a.data_root)],capture_output=True)
    state=State(a.source,capture_root); Handler.state=state; server=ThreadingHTTPServer((a.host,a.port),Handler); state.start(); signal.signal(signal.SIGINT,lambda *_:threading.Thread(target=server.shutdown).start()); print(f"[dataset-web] data root: {a.data_root}"); print(f"[dataset-web] open http://{a.host}:{a.port}")
    try: server.serve_forever()
    finally: state.shutdown(); server.server_close()
if __name__=="__main__": main()
