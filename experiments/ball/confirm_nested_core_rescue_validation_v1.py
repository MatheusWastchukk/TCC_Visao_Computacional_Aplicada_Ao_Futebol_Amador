"""One-shot VALIDATION confirmation of the policy frozen by nested TRAIN LOMO."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gap_add(chosen: list[dict[str, object]], ranked: list[dict[str, object]], limit: int) -> None:
    added = 0
    for row in ranked:
        center = float(row["delivery_start_seconds"]) + 20.0
        if any(str(row["group_uid"]) == str(other["group_uid"]) or abs(center - (float(other["delivery_start_seconds"]) + 20.0)) < 45.0 for other in chosen):
            continue
        chosen.append(row); added += 1
        if added >= limit:
            break


def measure(name: str, by_match: dict[str, list[dict[str, object]]], goals: dict[str, list[dict[str, str]]]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    per_match=[]; goal_detail=[]
    total_covered=total_goals=total_segments=0; total_seconds=0.0; total_candidates=0
    for match, rows in sorted(by_match.items()):
        source_ranks={}
        for source in ("rank_score","visual_score_goal","fusion_score_goal","net_motion"):
            order=sorted(rows,key=lambda r:(-float(r[source]),str(r["group_uid"])))
            source_ranks[source]={str(r["group_uid"]):i for i,r in enumerate(order,1)}
        rrf3=defaultdict(float)
        for source in ("rank_score","visual_score_goal","fusion_score_goal"):
            for uid,rank in source_ranks[source].items():rrf3[uid]+=1.0/(60+rank)
        if name=="temporal_top60_gap45":
            chosen=sorted(rows,key=lambda r:(-float(r["rank_score"]),str(r["group_uid"])))
            chosen=[];gap_add(chosen,sorted(rows,key=lambda r:(-float(r["rank_score"]),str(r["group_uid"]))),60)
        else:
            chosen=[]
            gap_add(chosen,sorted(rows,key=lambda r:(-float(r["rank_score"]),str(r["group_uid"]))),40)
            gap_add(chosen,sorted(rows,key=lambda r:(-rrf3[str(r["group_uid"])],str(r["group_uid"]))),10)
            gap_add(chosen,sorted(rows,key=lambda r:(-float(r["net_motion"]),str(r["group_uid"]))),20)
        intervals=[(float(r["delivery_start_seconds"]),float(r["delivery_end_seconds"])) for r in chosen]
        covered=0
        for goal in goals[match]:
            hit=any(a<=float(goal["event_timestamp_seconds"])<=b for a,b in intervals)
            covered+=int(hit)
            goal_detail.append({"method":name,"match_id":match,"goal_catalog_id":goal["goal_catalog_id"],"event_timestamp_seconds":goal["event_timestamp_seconds"],"covered":hit})
        merged=[]
        for a,b in sorted(intervals):
            if not merged or a>merged[-1][1]:merged.append([a,b])
            else:merged[-1][1]=max(merged[-1][1],b)
        minutes=sum(b-a for a,b in merged)/60.0
        per_match.append({"method":name,"match_id":match,"catalog_goals":len(goals[match]),"covered_goals":covered,"selected_candidates":len(chosen),"merged_segments":len(merged),"unique_review_minutes":f"{minutes:.3f}"})
        total_covered+=covered;total_goals+=len(goals[match]);total_candidates+=len(chosen);total_segments+=len(merged);total_seconds+=minutes*60
    summary={"method":name,"catalog_goals":total_goals,"covered_goals":total_covered,"recall":f"{total_covered/total_goals:.6f}","selected_candidates":total_candidates,"merged_segments":total_segments,"unique_review_minutes_per_match":f"{total_seconds/60/len(by_match):.3f}"}
    return summary,per_match,goal_detail


def main()->int:
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores",type=Path,default=root/"experiments/ball/TIERED-WINDOW-VALIDATION-CONFIRMATION-V1/validation_candidate_scores_net_motion_v1.csv")
    parser.add_argument("--goal-timestamps",type=Path,default=root/"experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-VALIDATION-V1/goal_catalog_timestamp_manifest_v1_annotated.csv")
    parser.add_argument("--output",type=Path,default=root/"experiments/ball/NESTED-CORE-RESCUE-VALIDATION-CONFIRMATION-V1")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    candidates=read_csv(args.candidate_scores);catalog=read_csv(args.goal_timestamps)
    if len(candidates)!=1944 or any(r.get("split","").lower()!="validation" for r in candidates):raise AssertionError("Expected 1,944 VALIDATION candidate rows only")
    if any(r.get("split","").lower()!="validation" for r in catalog):raise AssertionError("Only VALIDATION goal catalog is accepted")
    goals=defaultdict(list)
    for row in catalog:
        if row["annotation_status"]=="completed" and row.get("event_timestamp_seconds"):goals[row["match_id"]].append(row)
    by_match=defaultdict(list)
    for row in candidates:by_match[row["match_id"]].append(row)
    if set(by_match)!=set(goals) or len(by_match)!=8:raise ValueError("Candidate and annotated-goal matches do not match")
    summaries=[];per_match=[];coverage=[]
    for method in ("temporal_top60_gap45","nested_CORE40_RRF3_10_NET20"):
        summary,details,goal_rows=measure(method,by_match,goals)
        summaries.append(summary);per_match.extend(details);coverage.extend(goal_rows)
    goal_key={(r["match_id"],r["goal_catalog_id"]):r for r in coverage if r["method"]=="temporal_top60_gap45"}
    test_key={(r["match_id"],r["goal_catalog_id"]):r for r in coverage if r["method"]=="nested_CORE40_RRF3_10_NET20"}
    for row in coverage:
        key=(row["match_id"],row["goal_catalog_id"])
        row["gained_vs_temporal"]=(row["method"]=="nested_CORE40_RRF3_10_NET20" and row["covered"] and not goal_key[key]["covered"])
        row["lost_vs_temporal"]=(row["method"]=="nested_CORE40_RRF3_10_NET20" and goal_key[key]["covered"] and not row["covered"])
    gained=sum(r["covered"] and not goal_key[k]["covered"] for k,r in test_key.items())
    lost=sum(goal_key[k]["covered"] and not r["covered"] for k,r in test_key.items())
    args.output.mkdir(parents=True)
    write_csv(args.output/"validation_confirmation_summary_v1.csv",summaries)
    write_csv(args.output/"validation_confirmation_per_match_v1.csv",per_match)
    write_csv(args.output/"validation_confirmation_goal_coverage_v1.csv",coverage)
    inputs=[args.candidate_scores,args.goal_timestamps]
    meta={"experiment_id":args.output.name,"scope":"one-shot VALIDATION scoring; all model scores frozen from TRAIN-only fits; no tuning from these labels; TEST not accessed",
          "policy_frozen_from":"NESTED-CORE-RESCUE-LOMO-EXP-0001; CORE40 + RRF3 Top-10 rescue + net_motion Top-20 rescue; gap45; original 20+40-second review delivery windows",
          "reference":"temporal Top-60 with gap45",
          "gross_gains_vs_temporal":int(gained),"gross_losses_vs_temporal":int(lost),
          "validation_reuse_limitation":"The VALIDATION goal catalog had previously been inspected for the separate B4 confirmation. This A1 configuration and comparison were fixed from TRAIN before this scoring; treat as confirmatory evidence with shared-validation multiplicity, not as a pristine untouched batch.",
          "decision":"Interpret once; no tuning or V4 promotion from this result.",
          "inputs_sha256":{str(p.relative_to(root)):sha256(p) for p in inputs}}
    (args.output/"summary_v1.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"summary":summaries,"gross_gains":gained,"gross_losses":lost},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
