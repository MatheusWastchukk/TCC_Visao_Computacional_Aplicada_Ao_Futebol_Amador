"""Avalia a fusão entre candidato espacial e contexto pós-evento."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(description="Avalia GOAL-CLIP-VERIFIER com reinício pós-evento.")
    p.add_argument("--contexts",type=Path,required=True);p.add_argument("--candidate-files",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    with a.contexts.open(newline="",encoding="utf-8") as f: contexts=[r for r in csv.DictReader(f) if r["annotation_status"]=="completed"]
    candidates={}
    for path in a.candidate_files:
        with path.open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f): candidates[r["event_id"]]=r["decision"]=="POSSIVEL_GOL_CANDIDATO"
    rows=[]
    for row in contexts:
        spatial=candidates.get(row["event_id"],False); restart=row["context_decision"]=="RESTART_CONFIRMED"
        if spatial and restart: decision="POSSIBLE_GOAL"
        elif spatial and row["context_decision"]=="NO_RESTART_OBSERVED": decision="NO_GOAL_OR_INVALIDATED"
        elif not spatial and restart: decision="MISSED_SPATIAL_CANDIDATE"
        else: decision="NOT_EVALUABLE"
        rows.append({"context_id":row["context_id"],"event_id":row["event_id"],"expected_event_label":row["expected_event_label"],"match_id":row["match_id"],"spatial_candidate":str(spatial),"restart_confirmed":str(restart),"context_decision":row["context_decision"],"final_decision":decision})
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"goal_clip_verifier_results.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics={"experiment_id":a.output.name,"contexts":len(rows),"spatial_candidates":sum(r["spatial_candidate"]=="True" for r in rows),"possible_goal":sum(r["final_decision"]=="POSSIBLE_GOAL" for r in rows),"rejected_or_invalidated":sum(r["final_decision"]=="NO_GOAL_OR_INVALIDATED" for r in rows),"missed_spatial_candidate":sum(r["final_decision"]=="MISSED_SPATIAL_CANDIDATE" for r in rows),"note":"Piloto em contextos anotados manualmente; não representa execução completa de vídeo."}
    (a.output/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(metrics,ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
