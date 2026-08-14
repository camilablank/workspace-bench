# pending/ — cut item sets, kept with reasons

- `items_pilot_v1_4x4_5x5.json` — the first pilot bank (4x4 + 5x5, path len 5–9), ARCHIVED
  2026-08-13 after the two-condition behavioral gate failed it: 1/144 combined gate_pass,
  verbalize best cell 4/12 greedy (4x4 detour), passes_through at coin-flip. Failure was
  near-miss, not noise — verbalizations drop one cell or teleport once, count errors are
  modally +1 (cells-vs-moves fencepost, 16/48), 34/48 kth_step answers are on the gold path
  at the wrong index. Gate fields are embedded in the items; raw draws:
  `outputs/oracle_lens_evals/maze_path_eval/gate_raw_v1_4x4_5x5.json`. Superseded by the
  3x3/4x4 short-path v2 bank (same generator, SIZES/SIZE_LEN changed, count wording
  de-ambiguated, label alphabet drops A).
