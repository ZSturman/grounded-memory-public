"""Explicit setup downloads; caches live outside the repository."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

revisions = {}
for model in ["facebook/dinov2-small", "mlx-community/whisper-small-mlx"]:
    print("Preparing local model:", model, flush=True)
    revisions[model] = Path(
        snapshot_download(model, ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.h5"])
    ).name
checkpoint = hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt")
folder = Path(os.getenv("GM_DATA_DIR") or Path.home() / "Library/Application Support/Grounded Memory")
folder.mkdir(parents=True, exist_ok=True)
(folder / "models.json").write_text(
    __import__("json").dumps({"sam_checkpoint": checkpoint, "revisions": revisions}, indent=2)
)
print("Models prepared. SAM checkpoint:", checkpoint)
