#!/usr/bin/env python3
"""Build a browser review for two annotators' state partitions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="paper_experiments/annotations/raw")
    parser.add_argument("--agreement", default="paper_experiments/annotations/agreement_report.csv")
    parser.add_argument("--manifest", default="paper_experiments/manifests/foil_calibration.csv")
    parser.add_argument("--annotator-a", default="owner")
    parser.add_argument("--annotator-b", default="collaborator_a")
    parser.add_argument("--out", default="paper_experiments/paper_ready/calibration_review/index.html")
    args = parser.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        manifest = {row["clip_id"]: row for row in csv.DictReader(handle)}
    with open(args.agreement, newline="", encoding="utf-8") as handle:
        metrics = {row["clip_id"]: row for row in csv.DictReader(handle)}

    items = []
    raw_root = Path(args.raw_root)
    for clip_id in manifest:
        path_a = raw_root / args.annotator_a / f"{clip_id}.json"
        path_b = raw_root / args.annotator_b / f"{clip_id}.json"
        if not path_a.exists() or not path_b.exists():
            continue
        ann_a, ann_b = load_json(path_a), load_json(path_b)
        row = manifest[clip_id]
        items.append(
            {
                "clip_id": clip_id,
                "video": f"/paper_experiments/annotation_packages/foil_double_label_owner/videos/{Path(row['video_path']).name}",
                "fps": float(row["fps"]),
                "start": int(float(row["frame_start"])),
                "end": int(float(row["frame_end"])),
                "a": ann_a,
                "b": ann_b,
                "metrics": metrics.get(clip_id, {}),
            }
        )

    html = """<!doctype html><html><head><meta charset='utf-8'/><title>Calibration Review</title>
<style>
body{font-family:Arial,sans-serif;margin:16px;color:#18212a;background:#f5f7f8}h1{margin-bottom:4px}.intro{max-width:980px;line-height:1.45}.item{background:#fff;border:1px solid #d8dee3;padding:12px;margin:14px 0;border-radius:6px}.grid{display:grid;grid-template-columns:minmax(380px,600px) 1fr;gap:16px}video{width:100%;background:#000}.track{position:relative;height:38px;background:#dfe4e8;margin:6px 0;border:1px solid #c5cdd4}.seg{position:absolute;top:0;height:100%;color:#fff;font-size:11px;display:flex;align-items:center;justify-content:center;border-right:2px solid #fff;cursor:pointer;overflow:hidden}.stable{background:#24785a}.active{background:#d45318}.ignored{position:absolute;top:0;height:100%;background:#6b7280;color:#fff;font-size:10px;display:flex;align-items:center;justify-content:center}.label{font-size:12px;font-weight:bold;margin-top:8px}.meta{font-size:12px;color:#53616d}.flags{color:#a44200}.review{width:100%;min-height:55px}.decision{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:13px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>Calibration Review</h1><div class='intro'>Review disagreements without consulting model predictions. Focus on whether a visible change deserves a new state even when both neighboring states are active. Record one consensus partition and any rule clarification. Orange is active; green is stable.</div><div id='root'></div>
<script>const items=__ITEMS__;const A=__A__,B=__B__;const root=document.getElementById('root');
function flags(a){return Object.entries(a.quality_flags||{}).filter(([k,v])=>v&&k!=='valid').map(([k])=>k).join(', ')||'none'}
function endpoint(ann,item){return Number((ann.evaluation_range||ann.analyzed_range||{}).end_frame??item.end)}
function track(item,ann,video){const el=document.createElement('div');el.className='track';const n=item.end-item.start+1;(ann.segments||[]).forEach((s,i)=>{const d=document.createElement('div');d.className='seg '+s.activity;d.style.left=(100*(s.start_frame-item.start)/n)+'%';d.style.width=(100*(s.end_frame-s.start_frame+1)/n)+'%';d.textContent='S'+(i+1);d.title=`${s.start_frame}-${s.end_frame} ${s.activity}`;d.onclick=()=>{video.currentTime=s.start_frame/item.fps;video.play()};el.appendChild(d)});const end=endpoint(ann,item);if(end<item.end){const tail=document.createElement('div');tail.className='ignored';tail.style.left=(100*(end+1-item.start)/n)+'%';tail.style.width=(100*(item.end-end)/n)+'%';tail.textContent='excluded';el.appendChild(tail)}return el}
for(const item of items){const d=document.createElement('div');d.className='item';const m=item.metrics||{};d.innerHTML=`<h2>${item.clip_id}</h2><div class='meta'>Boundary F1 ±5: ${Number(m.boundary_f1_pm5||0).toFixed(3)} | Activity agreement: ${Number(m.activity_frame_agreement||0).toFixed(3)}</div><div class='grid'><div><video controls preload='metadata' src='${item.video}'></video></div><div class='parts'></div></div><div class='decision'><label><input type='checkbox'/> cuts agreed</label><label><input type='checkbox'/> activity agreed</label><label><input type='checkbox'/> endpoints agreed</label><label><input type='checkbox'/> exclude for pose quality</label></div><textarea class='review' placeholder='Consensus cuts, activity labels, endpoint, and any protocol clarification'></textarea>`;const video=d.querySelector('video'),parts=d.querySelector('.parts');for(const [name,ann] of [[A,item.a],[B,item.b]]){const l=document.createElement('div');l.className='label';l.textContent=`${name}: ${ann.segments.length} states | last included frame ${endpoint(ann,item)}`;parts.appendChild(l);parts.appendChild(track(item,ann,video));const f=document.createElement('div');f.className='meta'+(flags(ann)!=='none'?' flags':'');f.textContent='flags: '+flags(ann);parts.appendChild(f)}const key='calibration-review:'+item.clip_id,checks=[...d.querySelectorAll('.decision input')],note=d.querySelector('.review');let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}checks.forEach((box,i)=>box.checked=Boolean((saved.checks||[])[i]));note.value=saved.note||'';const persist=()=>localStorage.setItem(key,JSON.stringify({checks:checks.map(x=>x.checked),note:note.value}));checks.forEach(x=>x.onchange=persist);note.oninput=persist;root.appendChild(d)}
</script></body></html>"""
    html = html.replace("__ITEMS__", json.dumps(items)).replace("__A__", json.dumps(args.annotator_a)).replace("__B__", json.dumps(args.annotator_b))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} clips={len(items)}")


if __name__ == "__main__":
    main()
