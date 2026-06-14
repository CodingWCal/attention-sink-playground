"""Generate the real-token §01b 3-D tokenization widget into .threejs_proto/

Tokenizes a sample prompt with the gpt2 tokenizer, cleans the byte-level markers
(same as the notebook's clean_token), injects the tokens into the widget template,
and writes a standalone index.html so the preview tools can render the actual
tile animation (the headless marimo export can't show it).
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

ACCENT = "#2E6BDB"
PROMPT = "The quick brown fox jumps over the lazy dog."


def clean_token(t: str) -> str:
    return t.replace("Ġ", "␣").replace("Ċ", "⏎") or "∅"


tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok(PROMPT, return_tensors="pt")["input_ids"][0]
toks = tok.convert_ids_to_tokens(ids)
data = {
    "accent": ACCENT,
    "tokens": [{"text": clean_token(t), "first": i == 0} for i, t in enumerate(toks)],
}
print(f"{len(toks)} tokens: {[clean_token(t) for t in toks]}")

TEMPLATE = Path(__file__).with_name("_token3d_template.html").read_text(encoding="utf-8")
html = TEMPLATE.replace("__DATA__", json.dumps(data))
# expose stats for headless verification
html = html.replace(
    "    controls.update();\n    renderer.render(scene,camera);",
    "    controls.update();\n    renderer.render(scene,camera);\n"
    "    window.__rc=(window.__rc||0)+1;window.__n=n;window.__t=t;"
    "    window.__draws=renderer.info.render.calls;",
)
out_path = Path(__file__).parent.parent / ".threejs_proto" / "index.html"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"wrote {out_path}")
