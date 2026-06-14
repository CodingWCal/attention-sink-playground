"""Headless check for the §06 collapse scatter framing.

Reproduces the notebook's PCA projection + content-token domain on gpt2 and
reports how much of the frame the content tokens actually fill at a shallow vs a
deep layer. Before the fix the token-0 outlier blew out the axes and every
content token collapsed into ~0% of the frame (the "empty chart" bug).
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The quick brown fox jumps over the lazy dog."

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
inputs = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

hidden = torch.stack(out.hidden_states)[:, 0].float().numpy()  # (L+1, S, D)
Lp1, S, D = hidden.shape
n_layers = Lp1 - 1

# Normalize each token to a unit vector so the projection captures DIRECTION
# (cosine), matching the collapse the over-mixing meter measures — directionally
# similar tokens land on top of each other, so collapse reads as clumping.
hidden = hidden / np.linalg.norm(hidden, axis=-1, keepdims=True).clip(1e-8)

# Shared 2-D PCA frame, fit on content tokens across all layers.
fit = hidden[:, 1:, :].reshape(-1, D)
mean = fit.mean(axis=0)
_, _, Vt = np.linalg.svd(fit - mean, full_matrices=False)
proj = (hidden - mean) @ Vt[:2].T  # (L+1, S, 2)

# Center each layer's CONTENT cloud at the origin so we see the cloud's *size*
# (the collapse) rather than its drift across the frame. token0 keeps its offset
# from that centroid, so the sink visibly sits apart.
cent = proj.copy()
for Li in range(Lp1):
    c0 = proj[Li, 1:, :].mean(axis=0) if S > 1 else proj[Li, 0]
    cent[Li] = proj[Li] - c0

# Frame = max content half-extent across layers (symmetric), padded.
content = cent[:, 1:, :].reshape(-1, 2)
r = float(np.abs(content).max()) * 1.18 or 1.0
xr = yr = (-r, r)
fw = fh = 2 * r

print(f"prompt tokens: {S}  layers: {n_layers}")
print(f"frame  x:[{xr[0]:.2f},{xr[1]:.2f}]  y:[{yr[0]:.2f},{yr[1]:.2f}]")
for Li in (0, n_layers // 2, n_layers - 1):
    c = cent[Li + 1, 1:, :]  # centered content tokens at this layer
    sx = (c[:, 0].max() - c[:, 0].min()) / fw
    sy = (c[:, 1].max() - c[:, 1].min()) / fh
    t0 = cent[Li + 1, 0]
    print(
        f"layer {Li:2d}: content spread x={sx:5.1%} y={sy:5.1%} of frame | "
        f"token0 at ({t0[0]:6.2f},{t0[1]:6.2f})"
    )
print("OK — content should fill a healthy %% at shallow/mid layers and shrink with depth.")
