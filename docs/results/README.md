# Result provenance

This directory separates reproducible KSVQE results from values that only survive in the team presentation.

- `ksvqe_benchmarks.csv`: values checked against evaluation logs where available. The challenge-split KoNViD-1k log in the workspace ends after 16/1200 samples, so its reported SRCC/PLCC are retained from the consolidated experiment summary and explicitly marked.
- `team_distortion_demo.csv`: one-video web demo copied from the team presentation. These rows are qualitative examples, not a benchmark and not evidence of a universal trend.
- `brisque_svr_reported.csv`: BRISQUE + SVR values reported for a 117-video LIVE-VQC validation split in the team presentation. The BRISQUE implementation and its raw logs are not present in this repository.

All quality columns in the team demo use a larger-is-better display scale. The presentation states that the native BRISQUE distortion score was reversed before display.
