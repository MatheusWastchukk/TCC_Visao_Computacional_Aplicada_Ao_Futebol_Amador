"""Anota evidências contextuais pós-evento diretamente no vídeo."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2
from prepare_goal_context_exp0001 import FIELDS, fmt

FLAGS={"c":"center_visible","b":"ball_at_center","p":"players_reset","k":"kickoff_observed","s":"score_changed","l":"camera_lost","e":"end_of_half"}
def save(path:Path,rows:list[dict[str,str]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
def main()->int:
    p=argparse.ArgumentParser(description="Anota contexto pós-evento e reinício no centro.")
    p.add_argument("--contexts",type=Path,required=True);p.add_argument("--videos-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--pending-only",action="store_true");p.add_argument("--max-preview-width",type=int,default=1280);a=p.parse_args()
    with a.contexts.open(newline="",encoding="utf-8") as f: source=list(csv.DictReader(f))
    existing={}
    if a.output.exists():
        with a.output.open(newline="",encoding="utf-8") as f: existing={r["context_id"]:r for r in csv.DictReader(f)}
    rows=[{field:existing.get(r["context_id"],r).get(field,"") for field in FIELDS} for r in source]
    pending=[r for r in rows if not a.pending_only or r["annotation_status"]!="completed"]
    window="Contexto pós-evento";cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    for index,row in enumerate(pending,1):
        cap=cv2.VideoCapture(str(a.videos_dir/row["video_filename"]));
        if not cap.isOpened():raise FileNotFoundError(a.videos_dir/row["video_filename"])
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 30);first=round(float(row["search_start_seconds"])*fps);last=round(float(row["search_end_seconds"])*fps);current=first;playing=False
        try:
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES,current);ok,image=cap.read()
                if not ok: break
                sec=current/fps;state="RODANDO" if playing else "PAUSADO";flags=" ".join(k.upper() for k,v in FLAGS.items() if row.get(v)=="True") or "nenhum"
                cv2.rectangle(image,(8,8),(min(1250,image.shape[1]-8),74),(0,0,0),-1)
                cv2.putText(image,f"{index}/{len(pending)} | {row['expected_event_label']} | {row['match_id']} | {fmt(sec)} | {state}",(16,32),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                cv2.putText(image,f"Espaco tocar | A/seta esq -1s D/seta dir +1s | J/L +-5s | C centro B bola P jogadores K kickoff | R confirmou N sem restart U incerto | marcados: {flags}",(16,59),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,220,120),2)
                scale=min(1.,a.max_preview_width/image.shape[1]);cv2.imshow(window,cv2.resize(image,None,fx=scale,fy=scale) if scale!=1 else image)
                key=cv2.waitKeyEx(max(1,round(1000/fps)) if playing else 0)
                if key in {-1,255}:
                    if playing: current=min(last,current+1)
                    continue
                char=chr(key&0xff).lower()
                if char=="q" or key==27: save(a.output,rows);cv2.destroyAllWindows();return 0
                if key==32: playing=not playing
                elif char=="a" or key==2424832:current=max(first,current-round(fps));playing=False
                elif char=="d" or key==2555904:current=min(last,current+round(fps));playing=False
                elif char=="j":current=max(first,current-round(fps*5));playing=False
                elif char=="l":current=min(last,current+round(fps*5));playing=False
                elif char in FLAGS: row[FLAGS[char]]="True";row["restart_timestamp_seconds"]=f"{sec:.3f}" if char=="k" else row["restart_timestamp_seconds"]
                elif char in {"r","n","u"}: row["context_decision"]={"r":"RESTART_CONFIRMED","n":"NO_RESTART_OBSERVED","u":"NOT_EVALUABLE"}[char];row["annotation_status"]="completed";save(a.output,rows);break
        finally:cap.release()
    cv2.destroyAllWindows();print(f"Concluído: {a.output}");return 0
if __name__=="__main__":raise SystemExit(main())
