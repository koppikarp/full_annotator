# Historical source annotations

`annotation_app_2.1_zip/` contains the exact original CSV members preserved from the obsolete application ZIP before its removal. These are source history, not additional training rows. Current training reads the two top-level `annotated_*_ground_truth.csv` files.

The archived base CSV has 1,697 rows. The current CSV has 1,776; cell 1736, cycle 10 changed from C in the archive to A in the current annotations. The archived blob CSV has 382 rows, including cell 4179, cycle 3, which is absent from the current 381-row CSV. Do not silently combine these snapshots or replace the current labels.

The former `.bak` blob CSV is an exact 324-row subset of the archived blob CSV over every column it contains. The archived CSV therefore preserves that information too. Original detection bits and channel-correctness judgments must remain intact so true signal presence can be reconstructed.
