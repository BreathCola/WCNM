#!/usr/bin/env python3
"""Build versioned mask-hard two-hit caches from the fixed Stage C mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.mesh_intersector import MeshIntersector
from geometry.two_hit import CACHE_SCHEMA, masked_camera_rays, scatter_two_hits


DEBUG_STEMS = {"000000", "000039", "000040", "000041", "000075", "000110"}


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path):
    ply=PlyData.read(path); vertex=ply["vertex"].data; face=ply["face"].data
    vertices=np.stack([vertex[name] for name in ("x","y","z")],axis=1).astype(np.float32)
    faces=np.stack(face["vertex_indices"]).astype(np.int64)
    return vertices,faces


def write_points(path: Path, points: np.ndarray):
    data=np.empty(len(points),dtype=[("x","f4"),("y","f4"),("z","f4")])
    data["x"],data["y"],data["z"]=points.T
    PlyData([PlyElement.describe(data,"vertex")],text=False).write(path)


def depth_png(values,valid,lo,hi):
    norm=np.clip((values-lo)/max(hi-lo,1e-8),0,1)
    image=cv2.applyColorMap((norm*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    image[~valid]=0
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-output",required=True,type=Path)
    parser.add_argument("--audit",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();args.mesh_output=args.mesh_output.resolve();args.audit=args.audit.resolve();args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError(f"refusing to overwrite cache output: {args.output}")
    metadata=json.loads((args.mesh_output/"mesh_metadata.json").read_text())
    if not metadata.get("two_hit_ready"):raise ValueError("fixed mesh is not two-hit ready")
    mesh_path=args.mesh_output/"glass_mesh.ply";vertices,faces=load_mesh(mesh_path)
    raw=sorted((args.audit/"raw_views").glob("*.npz"))
    if len(raw)!=111:raise ValueError("expected 111 audited raw views")
    intersector=MeshIntersector(vertices,faces)
    args.output.mkdir(parents=True);cache_dir=args.output/"mesh_hits";cache_dir.mkdir();debug_dir=args.output/"debug";debug_dir.mkdir()
    rows=[];back_samples=[];debug_values={}
    for path in raw:
        stem=path.stem
        with np.load(path) as data:view={key:data[key] for key in data.files}
        origins,directions,linear=masked_camera_rays(view)
        near,far,counts=intersector.intersect(origins,directions)
        maps=scatter_two_hits(view["mask_hard"].shape,linear,near,far,counts,directions,origins)
        hard=np.asarray(view["mask_hard"],bool);eroded=np.asarray(view["mask_eroded"],bool)
        valid=maps["valid_two_hit"]
        row={
            "stem":stem,"mask_hard_pixels":int(hard.sum()),"mask_eroded_pixels":int(eroded.sum()),
            "valid_two_hit_pixels":int(valid.sum()),
            "valid_fraction_hard":float(valid.sum()/max(hard.sum(),1)),
            "valid_fraction_eroded":float((valid&eroded).sum()/max(eroded.sum(),1)),
            "far_gt_near_fraction":float(np.mean(maps["t_far"][valid]>maps["t_near"][valid])) if valid.any() else 0.0,
            "hit_count_gt2_fraction":float(np.mean(maps["hit_count"][valid]>2)) if valid.any() else 0.0,
        };rows.append(row)
        payload={
            "schema":np.array(CACHE_SCHEMA),"stem":np.array(stem),
            "mesh_sha256":np.array(sha256_file(mesh_path)),
            "checkpoint_sha256":np.array(metadata["checkpoint_sha256"]),
            **maps,
        }
        np.savez_compressed(cache_dir/f"{stem}.npz",**payload)
        back_samples.append(maps["back_position"][valid][::16])
        if stem in DEBUG_STEMS:debug_values[stem]=maps
    near_samples=np.concatenate([m["t_near"][m["valid_two_hit"]][::16] for m in debug_values.values()])
    far_samples=np.concatenate([m["t_far"][m["valid_two_hit"]][::16] for m in debug_values.values()])
    lo=float(np.quantile(near_samples,.01));hi=float(np.quantile(far_samples,.99))
    for stem,maps in debug_values.items():
        directory=debug_dir/stem;directory.mkdir()
        Image.fromarray(depth_png(maps["t_near"],maps["valid_two_hit"],lo,hi)).save(directory/"near_depth.png")
        Image.fromarray(depth_png(maps["t_far"],maps["valid_two_hit"],lo,hi)).save(directory/"far_depth.png")
        Image.fromarray((maps["valid_two_hit"].astype(np.uint8)*255)).save(directory/"two_hit_valid.png")
    points=np.concatenate(back_samples);voxel=np.floor(points/0.01).astype(np.int64);_,unique=np.unique(voxel,axis=0,return_index=True);points=points[unique]
    write_points(args.output/"back_surface_points.ply",points)
    # Required canonical debug files use the explicitly reviewed 000039 view.
    for name in ("near_depth.png","far_depth.png","two_hit_valid.png"):
        (args.output/name).write_bytes((debug_dir/"000039"/name).read_bytes())
    report={
        "schema":CACHE_SCHEMA,"mesh":str(mesh_path),"mesh_sha256":sha256_file(mesh_path),
        "checkpoint_sha256":metadata["checkpoint_sha256"],"cache_count":len(rows),
        "ray_domain":"mask_hard only; Reflection ray domain unchanged",
        "debug_depth_scale":{"p01_near":lo,"p99_far":hi,"shared":True},"per_view":rows,
        "aggregate":{
            "valid_fraction_hard_mean":float(np.mean([r["valid_fraction_hard"] for r in rows])),
            "valid_fraction_hard_min":float(np.min([r["valid_fraction_hard"] for r in rows])),
            "valid_fraction_eroded_mean":float(np.mean([r["valid_fraction_eroded"] for r in rows])),
            "far_gt_near_fraction":float(np.mean([r["far_gt_near_fraction"] for r in rows])),
            "hit_count_gt2_fraction_mean":float(np.mean([r["hit_count_gt2_fraction"] for r in rows])),
        },
    }
    temporary=args.output/"cache_metadata.json.tmp";temporary.write_text(json.dumps(report,indent=2,sort_keys=True));os.replace(temporary,args.output/"cache_metadata.json")
    print(json.dumps(report["aggregate"],indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
