"""Day-1 de-risk: confirm eager attention returns real attention tensors.

Loads distilgpt2 with attn_implementation="eager", runs one prompt, and
verifies every assumption the notebook's metrics depend on (PRD sections
11, 18, 25).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"
PROMPT = "The quick brown fox jumps over the lazy dog."

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    attn_implementation="eager",  # REQUIRED: SDPA/Flash return None attentions
    dtype=torch.float32,
).to(device)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs, output_attentions=True, output_hidden_states=True)

assert out.attentions is not None, "attentions are None — eager attn not in effect"
assert all(a is not None for a in out.attentions), "a layer returned None attention"

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
n_layers = len(out.attentions)
n_heads = out.attentions[0].shape[1]
seq = out.attentions[0].shape[-1]
print(f"layers: {n_layers}, heads: {n_heads}, seq len: {seq}")
print(f"tokens: {tokens}")

# rows must sum to 1 (softmax over keys)
A = out.attentions[0][0, 0]
assert torch.allclose(A.sum(dim=-1), torch.ones(seq), atol=1e-4), "rows don't sum to 1"

# first-token sink metric, excluding the trivial query row 0
layerwise = [out.attentions[L][0, :, 1:, 0].mean().item() for L in range(n_layers)]
print("layer-wise avg attention to token 0 (query row 0 excluded):")
for L, v in enumerate(layerwise):
    bar = "#" * int(v * 60)
    print(f"  layer {L}: {v:6.1%}  {bar}")

# hidden states present for the over-mixing meter
assert out.hidden_states is not None and len(out.hidden_states) == n_layers + 1
print(f"hidden states: {len(out.hidden_states)} x {tuple(out.hidden_states[0].shape)}")

print("\nALL CHECKS PASSED — eager attention works, sink is measurable.")
