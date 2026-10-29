"""Compare original and cleaned loaders on fixed synthetic windows."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from lob_xai.checkpoints import load_ours_model
from lob_xai.gradcam import TimeSeriesGradCAM

torch.set_num_threads(2)
torch.manual_seed(17)
x = torch.tensor(np.random.default_rng(17).normal(size=(2, 100, 40)), dtype=torch.float32)
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
results = {}
for key, row in manifest["checkpoints"].items():
    model, _, _ = load_ours_model(str(root / row["path"]), device="cpu", strict=True)
    model.eval()
    with torch.no_grad():
        results[key + "_logits"] = model(x).numpy()
    for branch in range(4):
        cam = TimeSeriesGradCAM(model, "cpu", method="hirescam")
        try:
            signed, absolute = cam.generate_batch(x, branch_idx=branch, target_class=1)
            results[f"{key}_branch{branch}_signed"] = signed
            results[f"{key}_branch{branch}_absolute"] = absolute
        finally:
            cam.cleanup()
np.savez(sys.argv[3], **results)
print(f"Saved {len(results)} arrays from {len(manifest['checkpoints'])} checkpoints.")
