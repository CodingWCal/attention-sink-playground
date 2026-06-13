"""Render a static hero heatmap for the README (assets/heatmap.png).

Mirrors the notebook's layer-9/head-0 view on the default prompt, themed to
match the Claude Design mockup (blues scale, accent on the first column).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer

ACCENT = "#2E6BDB"
INK = "#14181F"
MUTED = "#7A828D"

PROMPT = "The quick brown fox jumps over the lazy dog."
LAYER, HEAD = 9, 0

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
inputs = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

A = out.attentions[LAYER][0, HEAD].float().numpy()
tokens = [t.replace("Ġ", "␣") for t in tok.convert_ids_to_tokens(inputs["input_ids"][0])]

fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max())

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9, family="monospace")
ax.set_yticklabels(tokens, fontsize=9, family="monospace")
ax.set_xlabel("key  (attended TO)", fontsize=10, color=MUTED, family="monospace")
ax.set_ylabel("query  (attending FROM)", fontsize=10, color=MUTED, family="monospace")
ax.set_title(
    f"Attention sink on token 0  ·  gpt2  ·  layer {LAYER}, head {HEAD}",
    fontsize=13, color=INK, pad=12,
)

# Highlight the first-token column (the sink).
ax.add_patch(
    Rectangle((-0.5, -0.5), 1, len(tokens), fill=False, edgecolor=ACCENT, lw=2.5)
)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, colors=MUTED)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("attention weight", fontsize=9, color=MUTED, family="monospace")

fig.tight_layout()
out_path = Path(__file__).resolve().parents[1] / "assets" / "heatmap.png"
fig.savefig(out_path, bbox_inches="tight", facecolor="white")
print(f"wrote {out_path}")
