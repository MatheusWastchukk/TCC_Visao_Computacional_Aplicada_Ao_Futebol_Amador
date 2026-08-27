# Pendências antes da execução temporal

A tabela possui 14 linhas preenchidas; falta recuperar a sequência 009:

```text
sequence_id: temporal_v10_candidate_009
match_id: match_12
video_path: data\Videos_Edited\Match_012.mp4
anchor_frame_id: match_12_random_02
start_seconds: 4228.760
end_seconds: 4232.760
start_time_hhmmss: 01:10:29
end_time_hhmmss: 01:10:33
anchor_label: not_goal
```

Também revisar o `true_label` destas linhas conforme as próprias observações indicam visibilidade parcial:

- `temporal_v10_candidate_011`: considerar `inconclusive`;
- `temporal_v10_candidate_012`: considerar `inconclusive`;
- `temporal_v10_candidate_015`: considerar `inconclusive`.

Após a correção, a tabela deve conter 15 linhas, todas com `annotation_status=completed`, e somente os rótulos `goal_structure`, `not_goal` ou `inconclusive`.
