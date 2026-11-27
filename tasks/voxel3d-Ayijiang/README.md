# Ayijiang: 3D voxel reconstruction

This contribution studies **Constraint-Preserving Structured Generation (CPSG)** for gravity-constrained HypoSpace voxel reconstruction. The final packaged method combines column-height generation with support projection, range-feedback retry, and duplicate control.

The original [voxel3d_for_github_tasks.zip](voxel3d_for_github_tasks.zip) is retained **byte-for-byte unchanged**. It has not been extracted into the repository.

## Inside the package

The archive contains `tasks/voxel3d/cpsg/` with:

- `README.md` and `REPRODUCIBILITY.md`: method, reported metrics, environment, and run commands.
- `src/methods/cpsg.py`: structured prompts, parsing, support projection, and decoding.
- `scripts/`: official evaluation and representation-equivalence audit.
- `configs/obs_pool_50.json`: fixed observation pool.
- `reported-results/` and `docs/`: saved summaries, audits, and final reports.
- `upstream/3d/`: packaged upstream task code and dataset, with attribution.

The original paths inside the ZIP are intentionally preserved, so its own reproduction instructions continue to describe the packaged layout. See the README inside the archive for the 50-observation, three-seed comparison and full metric definitions.
