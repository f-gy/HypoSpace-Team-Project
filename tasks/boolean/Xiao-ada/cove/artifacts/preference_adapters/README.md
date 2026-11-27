# Matched continuation adapters

Both checkpoints start from the original canonical-coverage adapter and add 80 updates. The coverage_continuation arm retains the coverage/SFT objective. The canonical_preference arm adds a class-level preference objective with training-only negative examples.

Validation selected coverage_continuation with BACE. The preference arm remains a reported control and did not win selection. Both checkpoints contain 311,296 trainable LoRA parameters. Hashes, training times and unchanged-base verification appear in manifest.json and results/canonical_preference_v5/.

The restore utility places these released checkpoints and the exact training targets in the local run directory. Restoration does not retrain the model.
