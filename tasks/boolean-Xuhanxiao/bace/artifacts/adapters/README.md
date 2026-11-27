# Original matched LoRA checkpoints

The three adapters contain the released balanced-SFT, canonical-coverage and string-coverage LoRA weights. Each follows the same 80-update warmup and 120-update continuation schedule. The frozen base model is pinned in model.local_mlx.lock.json.

Adapter tensors and configurations retain their original hashes. These files contain no base-model weights. Apache 2.0 and base-model licenses accompany the adapters. The restore utility validates the manifest before placing checkpoints in the local run directory.
