"""Headless logic check for the model-size comparison (notebook section 08).

Confirms the 'bigger/deeper models sink harder' data is sane: computes the
layer-wise sink curve for DistilGPT-2 (6L) and GPT-2 (12L) on the same prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The quick brown fox jumps over the lazy dog."

for name in ["distilgpt2", "gpt2"]:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="eager"
    ).eval()
    inputs = tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    attn = torch.stack(out.attentions)[:, 0].float()  # (L, H, S, S)
    Lc = attn.shape[0]
    curve = [attn[Li, :, 1:, 0].mean().item() for Li in range(Lc)]
    deep_mean = sum(curve[Lc // 2:]) / len(curve[Lc // 2:])  # deeper-half avg
    print(f"{name:11s}  {Lc:2d} layers   peak={max(curve):.0%}   deep-half avg={deep_mean:.0%}")
    print("   curve: " + "  ".join(f"{v:.0%}" for v in curve))
