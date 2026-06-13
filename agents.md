The deliverable is a marimo notebook: ONE Python file, notebook.py.
It is NOT a web app. design/ is a VISUAL REFERENCE ONLY (HTML/CSS from Claude
Design). Match its layout, colors, type, and section flow, but implement
everything with marimo (mo.ui.*, mo.md, mo.hstack/vstack), never React/HTML.
Spec is docs/PRD.md (sections 8, 11, 13, 15, 16).
Hard rules:
- Load models with attn_implementation="eager" or attentions return None.
- Gate inference behind mo.ui.run_button; cache the model; moving a slider
  must NOT re-run inference.
- Exclude query position 0 from first-token attention summaries (causal mask
  makes it trivially 100%).
- Default to ungated models (gpt2, distilgpt2, pythia). No Llama/Gemma default.
- Stay in scope: interpretability + viz on a pretrained model. No training.