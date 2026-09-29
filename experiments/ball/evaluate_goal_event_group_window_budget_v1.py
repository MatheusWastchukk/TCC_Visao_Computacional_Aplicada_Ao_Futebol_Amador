"""Avalia entrega por limites reais do grupo temporal, somente em TRAIN."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import cv2

def read(path):
    with Path(path).open(newline="", encoding="utf-8") as file: return list(csv.DictReader(file))

def windows(values):
    output=[]
    for value in values:
        before, after=(float(part) for part in value.split(":",1))
        if before<0 or after<=0: raise ValueError("Margens inválidas.")
        output.append((before,after))
    return sorted(set(output))

def groups(files):
    hits=defaultdict(list)
    for path in files:
        for row in read(path): hits[row["match_id"]].append(float(row["timestamp_seconds"]))
    output={}
    for match,times in hits.items():
        blocks=[]
        for timestamp in sorted(times):
            if not blocks or timestamp-blocks[-1][-1]>3.0: blocks.append([timestamp])
            else: blocks[-1].append(timestamp)
        for index,block in enumerate((item for item in blocks if len(item)>=2),1):
            output[f"{match}::temporal_group_{index:04d}"]=(block[0],block[-1])
    return output

def select(rows):
    chosen=[]
    for row in sorted(rows,key=lambda item:float(item["rank_score"]),reverse=True):
        if all(abs(float(row["best_timestamp_seconds"])-float(other["best_timestamp_seconds"]))>=45 for other in chosen): chosen.append(row)
    return chosen

def merge(intervals):
    output=[]
    for start,end in sorted(intervals):
        if not output or start>output[-1][1]: output.append([start,end])
        else: output[-1][1]=max(output[-1][1],end)
    return output

def duration(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(path)
    try: fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); frames=float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally: cap.release()
    if fps<=0 or frames<=0: raise ValueError(f"Metadados inválidos: {path}")
    return frames/fps

def main():
    parser=argparse.ArgumentParser(description="Caracteriza orçamento de janelas baseadas em grupos TRAIN.")
    parser.add_argument("--rank-predictions",type=Path,required=True); parser.add_argument("--goal-timestamps",type=Path,required=True)
    parser.add_argument("--hit-files",type=Path,nargs="+",required=True); parser.add_argument("--videos-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--candidate-limits",type=int,nargs="+",default=[20,30,40,50,60,70,80,90,100,999999])
    parser.add_argument("--margins",nargs="+",default=["5:10","5:15","10:10","10:15","10:20","15:15"])
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"A saída já existe: {args.output}")
    predictions=[row for row in read(args.rank_predictions) if row.get("model")=="random_forest"]
    bounds=groups(args.hit_files)
    if set(row["group_uid"] for row in predictions)!=set(bounds): raise ValueError("Grupos reconstruídos divergem das predições LOMO.")
    catalog=read(args.goal_timestamps)
    if any(row["split"]!="train" for row in catalog): raise AssertionError("Somente TRAIN.")
    goals=defaultdict(list); names={}
    for row in catalog:
        names[row["match_id"]]=row["video_filename"]
        if row["annotation_status"]=="completed" and row["event_timestamp_seconds"]: goals[row["match_id"]].append(float(row["event_timestamp_seconds"]))
    ranked=defaultdict(list)
    for row in predictions: ranked[row["match_id"]].append(row)
    diverse={match:select(rows) for match,rows in ranked.items()}; lengths={match:duration(args.videos_dir/names[match]) for match in ranked}
    total_goals=sum(map(len,goals.values())); rows=[]; max_available=max(map(len,diverse.values()))
    for before,after in windows(args.margins):
        for limit in sorted({min(value,max_available) for value in args.candidate_limits}):
            chosen={match:items[:limit] for match,items in diverse.items()}; merged={}; covered=0; raw=0
            for match,items in chosen.items():
                intervals=[]
                for row in items:
                    start,end=bounds[row["group_uid"]]; interval=(max(0,start-before),min(lengths[match],end+after)); intervals.append(interval); raw+=interval[1]-interval[0]
                merged[match]=merge(intervals)
                covered+=sum(any(start<=goal<=end for start,end in merged[match]) for goal in goals[match])
            unique=sum(end-start for items in merged.values() for start,end in items)
            rows.append({"margin_before_seconds":before,"margin_after_seconds":after,"k_limit":"pool_completo" if limit==max_available else limit,"catalog_goals":total_goals,"covered_goals":covered,"recall":f"{covered/total_goals:.6f}","candidates":sum(map(len,chosen.values())),"merged_segments":sum(map(len,merged.values())),"raw_review_seconds":f"{raw:.3f}","unique_review_seconds":f"{unique:.3f}","unique_review_minutes_per_match":f"{unique/60/len(chosen):.3f}"})
    args.output.mkdir(parents=True)
    with (args.output/"group_window_budget_curve_v1.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"experiment_id":args.output.name,"scope":"somente TRAIN; grupos reconstruídos e predições LOMO congeladas","catalog_goals":total_goals,"margins":[f"{a:g}+{b:g}" for a,b in windows(args.margins)],"warning":"Não treina modelo, não altera V4 e não usa VALIDATION/TEST."}
    (args.output/"summary_v1.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
