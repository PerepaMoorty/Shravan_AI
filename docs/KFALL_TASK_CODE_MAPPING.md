# KFall task code → direction mapping

The pipeline does **not** hardcode "task code F09 = forward fall" as a fact,
because the exact task-code-to-description mapping needs to be checked
against whatever KFall release you actually have (label sheets have varied
slightly across dataset versions). Instead:

1. `shravan_ai/direction.py` classifies each trial's direction primarily
   from the label file's own `Description` text (keyword match on
   forward/backward/lateral), since that field is already in the real
   label files and is more trustworthy than a memorized code table.
2. Task codes starting with `D` (ADL, non-fall) always map to
   `normal_only` regardless of description.
3. Trials whose description doesn't contain a recognizable direction
   keyword are left as `unknown_direction` and listed in
   `output/trial_timelines/_needs_review.csv` after a pipeline run —
   they are never silently guessed.
4. To force a specific task code to a direction (e.g. once you've checked
   the real KFall paper's task table, or for a description your keyword
   list doesn't catch), add a row to `shravan_ai/task_code_map.csv`.

**Action item before using real KFall data:** open a couple of the actual
label files, skim the `Description` column, and confirm the keyword list
in `direction.py` (`forward` / `backward` / `lateral`, `left`, `right`,
`side`, `sideway`) covers the wording used. Add overrides to
`task_code_map.csv` for anything that doesn't match cleanly (e.g. fainting
falls, which aren't really directional and may need their own bucket).

## What real KFall label sheets actually look like (verified)

Checked against the real dataset directly — three things `pipeline.py`
handles that aren't obvious from the column names alone:

- **`.xlsx`, not `.csv`.** `Task Code (Task ID)` and `Description` are only
  populated on each fall's *first* row — a leftover of Excel merged cells.
  `pipeline.py` forward-fills both columns before reading rows.
- **The visible task code isn't the sensor filename's task number.**
  `F01 (20)` means task `F01`, sensor files `S06T20R0*.csv` — the number
  in parentheses is what matters for locating the file; the sheet's
  `Trial ID` column is just the repetition number (1-5), not a filename
  fragment.
- **No ADL rows at all.** A real subject's label sheet lists only
  `F01`-`F15`. Every other sensor recording in that subject's folder
  (walking, sitting, etc.) has no label row whatsoever — `pipeline.py`
  treats any unclaimed sensor file as an ADL (`task_code=D00`,
  `normal_only`) rather than skipping it.

Verified end-to-end against subject `SA06`: 156 sensor files, 156 trials
processed, 0 skipped, 4 flagged for manual review (all the same
non-directional "fainting" fall type, correctly not guessed).
