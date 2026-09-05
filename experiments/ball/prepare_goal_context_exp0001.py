"""Prepara janelas pós-evento para o piloto de contexto de gol."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

FIELDS = ["context_id","event_id","expected_event_label","match_id","video_filename","event_timestamp_seconds","search_start_seconds","search_end_seconds","search_start_hhmmss","search_end_hhmmss","center_visible","ball_at_center","players_reset","kickoff_observed","score_changed","camera_lost","end_of_half","restart_timestamp_seconds","context_decision","annotation_status","notes"]
def fmt(value: float) -> str:
    h, rest = divmod(max(0, round(value)), 3600); m, s = divmod(rest, 60); return f"{h:02d}:{m:02d}:{s:02d}"
def load_events(events: Path, moments: Path, label: str) -> list[dict[str,str]]:
    with events.open(newline="", encoding="utf-8") as file: by_id={r["event_id"]:r for r in csv.DictReader(file) if r["event_label"]==label}
    with moments.open(newline="", encoding="utf-8") as file: return [{**by_id[r["event_id"]], **r} for r in csv.DictReader(file) if r["event_id"] in by_id and r["annotation_status"]=="completed"]
def main() -> int:
    p=argparse.ArgumentParser(description="Prepara GOAL-CONTEXT-EXP-0001.")
    p.add_argument("--v2-events",type=Path,required=True);p.add_argument("--v2-moments",type=Path,required=True);p.add_argument("--v3-events",type=Path,required=True);p.add_argument("--v3-moments",type=Path,required=True);p.add_argument("--negative-events",type=Path,required=True);p.add_argument("--negative-moments",type=Path,required=True);p.add_argument("--negative-event-id",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--start-after",type=float,default=10);p.add_argument("--window",type=float,default=120);a=p.parse_args()
    positives=load_events(a.v2_events,a.v2_moments,"POSSIVEL_GOL")+load_events(a.v3_events,a.v3_moments,"POSSIVEL_GOL")
    negatives=[event for event in load_events(a.negative_events,a.negative_moments,"NAO_GOL") if event["event_id"]==a.negative_event_id]
    if len(negatives)!=1: raise ValueError("O negativo representativo informado não foi encontrado.")
    rows=[]
    for i,event in enumerate(positives+negatives,1):
        t=float(event["event_timestamp_seconds"]); start=t+a.start_after; end=start+a.window
        rows.append({"context_id":f"goal_context_{i:03d}","event_id":event["event_id"],"expected_event_label":event["event_label"],"match_id":event["match_id"],"video_filename":event["video_filename"],"event_timestamp_seconds":f"{t:.3f}","search_start_seconds":f"{start:.3f}","search_end_seconds":f"{end:.3f}","search_start_hhmmss":fmt(start),"search_end_hhmmss":fmt(end),"center_visible":"","ball_at_center":"","players_reset":"","kickoff_observed":"","score_changed":"","camera_lost":"","end_of_half":"","restart_timestamp_seconds":"","context_decision":"","annotation_status":"pending","notes":"Piloto manual de contexto pós-evento; não usar como verdade oficial."})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print(f"contextos={len(rows)}\ncsv={a.output}");return 0
if __name__=="__main__": raise SystemExit(main())
