"""Generate the real-data §03b 3-D attention-flow widget into .threejs_proto/

Computes, per layer, each token's attention to token 0 (averaged over heads) on a
sample prompt, injects it into the flow widget template, and writes a standalone
index.html so the preview tools can render the actual arcs + layer animation.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACCENT = "#2E6BDB"
PROMPT = "The quick brown fox jumps over the lazy dog."
CAP = 24  # cap tokens shown so the arcs stay legible


def clean_token(t: str) -> str:
    return t.replace("Ġ", "␣").replace("Ċ", "⏎") or "∅"


tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
inputs = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

attn = torch.stack(out.attentions)[:, 0].float()        # (L, H, S, S)
toks = tok.convert_ids_to_tokens(inputs["input_ids"][0])
disp = min(len(toks), CAP)
L = attn.shape[0]
w = [[float(attn[li, :, q, 0].mean()) for q in range(disp)] for li in range(L)]
data = {"accent": ACCENT, "tokens": [clean_token(t) for t in toks[:disp]], "w": w}
print(f"{disp} tokens, {L} layers; token attn->0 at layer 0: "
      f"{[round(x,2) for x in w[0]]}")
print(f"... at last layer: {[round(x,2) for x in w[-1]]}")

TEMPLATE = Path(__file__).with_name("_flow3d_template.html").read_text(encoding="utf-8")
html = TEMPLATE.replace("__DATA__", json.dumps(data))
html = html.replace(
    "    controls.update();\n    renderer.render(scene,camera);",
    "    controls.update();\n    renderer.render(scene,camera);\n"
    "    window.__rc=(window.__rc||0)+1;window.__arcs=arcs.length;window.__L=L;"
    "    window.__auto=controls.autoRotate;window.__set=setLayer;"
    "    window.__opNow=()=>arcs.map(a=>+a.mat.opacity.toFixed(3));",
)
out_path = Path(__file__).parent.parent / ".threejs_proto" / "index.html"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"wrote {out_path}")
