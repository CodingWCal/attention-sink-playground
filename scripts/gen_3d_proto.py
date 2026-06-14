"""Generate the real-data three.js widget HTML and write it to .threejs_proto/

Reproduces the notebook's §06b 3-D data cell on gpt2, injects it into the SAME
widget template, and writes a standalone index.html so the preview tools can
render the actual visualization (real token cloud, real collapse) — the part the
headless marimo export can't show.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACCENT = "#2E6BDB"
PROMPT = "The quick brown fox jumps over the lazy dog."

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
inputs = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

H = torch.stack(out.hidden_states)[:, 0].float().numpy()
H = H / np.linalg.norm(H, axis=-1, keepdims=True).clip(1e-8)
Lp1, S, D = H.shape
n_layers = Lp1 - 1
fit = H[:, 1:, :].reshape(-1, D)
mean = fit.mean(axis=0)
_, _, Vt = np.linalg.svd(fit - mean, full_matrices=False)
proj = (H - mean) @ Vt[:3].T
cl = np.stack([proj[Li + 1] - proj[Li + 1, 1:].mean(axis=0) for Li in range(n_layers)])
scale = np.abs(cl[:, 1:, :]).max(axis=(0, 1)).clip(1e-6)
norm = np.clip(cl / scale * 0.9, -1.0, 1.0)
data = {
    "accent": ACCENT,
    "layers": int(n_layers),
    "points": [
        [
            {"x": round(float(norm[Li, ti, 0]), 4),
             "y": round(float(norm[Li, ti, 1]), 4),
             "z": round(float(norm[Li, ti, 2]), 4),
             "first": ti == 0}
            for ti in range(S)
        ]
        for Li in range(n_layers)
    ],
}
print(f"tokens={S} layers={n_layers}")
for Li in (0, n_layers // 2, n_layers - 1):
    c = norm[Li + 1 - 1, 1:, :] if False else norm[Li, 1:, :]
    rng = float(np.abs(c).max())
    print(f"layer {Li:2d}: content max|coord|={rng:.3f}  (smaller = collapsed)")

TEMPLATE = Path(__file__).with_name("_widget_template.html").read_text(encoding="utf-8")
html = TEMPLATE.replace("__DATA__", json.dumps(data))
# expose render/camera stats for headless interaction verification
html = html.replace(
    "    controls.update();\n    renderer.render(scene,camera);",
    "    controls.update();\n    renderer.render(scene,camera);\n"
    "    window.__camPos=camera.position.toArray();window.__n=spheres.length;window.__t=target;",
)
out_path = Path(__file__).parent.parent / ".threejs_proto" / "index.html"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"wrote {out_path}")
