# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "altair",
#     "pandas",
#     "numpy",
# ]
# ///
"""Attention Sink Playground — an interactive marimo notebook.

Visualizes why LLMs pour attention onto their first token (Barbero et al. 2025,
arXiv:2504.02732). Reactive architecture per the PRD §11:
  Cell A  load model (cached, runs once)
  Cell B  run inference (gated behind Run / preset change)
  Cell C  compute metrics (cheap, reactive to layer/head)
  Cell D  render visuals (cheap, reactive)
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Attention Sink Playground")


# ============================================================ imports
@app.cell
def _():
    import functools

    import altair as alt
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import marimo as mo

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        alt,
        functools,
        mo,
        np,
        pd,
        torch,
    )


# ============================================================ design tokens
@app.cell
def _(alt):
    # Carried over from the Claude Design mockup (design/*.dc.html).
    ACCENT = "#2E6BDB"      # "the sink" — the protagonist colour, used consistently
    INK = "#14181F"
    MUTED = "#7A828D"
    SUBTLE = "#4E5663"
    BORDER = "#E2E8F0"
    CARD = "#FFFFFF"
    PAGE = "#F8FAFC"
    GOOD = "#2E8A5B"
    MONO = "IBM Plex Mono, ui-monospace, monospace"
    SERIF = "Newsreader, Georgia, serif"

    def styled(chart, h=320, frame=False):
        """Apply the shared chart theme. width='container' makes the chart fill
        the page width (the notebook runs full-width), so visuals read large.
        frame=True draws a border around the plot — used for the abstract PCA
        scatter so the dots sit in a defined box, not floating in white space."""
        return (
            chart.properties(height=h, width="container", background="transparent")
            .configure_view(strokeWidth=1 if frame else 0, stroke=BORDER)
            .configure_axis(
                labelFont=MONO,
                titleFont=MONO,
                labelColor=MUTED,
                titleColor=SUBTLE,
                labelFontSize=11,
                titleFontSize=12,
                domainColor=BORDER,
                tickColor=BORDER,
                grid=False,
            )
            .configure_legend(
                labelFont=MONO,
                titleFont=MONO,
                labelColor=MUTED,
                titleColor=SUBTLE,
                labelFontSize=11,
                titleFontSize=12,
            )
        )

    return ACCENT, BORDER, CARD, GOOD, INK, MONO, MUTED, PAGE, SERIF, SUBTLE, styled


# ============================================================ Cell A — model (cached)
@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, functools, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @functools.cache
    def load_model(name: str):
        """Load tokenizer + model once. Cached so layer/head changes never reload."""
        tok = AutoTokenizer.from_pretrained(name)
        # attn_implementation="eager" is REQUIRED — SDPA/Flash return None
        # attentions. On GPU use bf16; transformers renamed torch_dtype -> dtype
        # in v5, so try both. On CPU keep the float32 default, which works across
        # transformers versions (molab may ship either v4 or v5).
        kwargs = {"attn_implementation": "eager"}
        if device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    name, dtype=torch.bfloat16, **kwargs
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=torch.bfloat16, **kwargs
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        return tok, model.to(device).eval()

    @functools.cache
    def run_inference(prompt: str, name: str = "gpt2"):
        """Cell B — the only expensive step per prompt. Cached by (prompt, model)."""
        tok, model = load_model(name)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(
                **inputs, output_attentions=True, output_hidden_states=True
            )
        # attentions: tuple[L] of (1, H, S, S)  ->  (L, H, S, S)
        attn = torch.stack(out.attentions)[:, 0].float().cpu()
        # hidden_states: tuple[L+1] of (1, S, D)  ->  (L+1, S, D)
        hidden = torch.stack(out.hidden_states)[:, 0].float().cpu()
        tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
        return {"attn": attn, "hidden": hidden, "tokens": tokens}

    return device, load_model, run_inference


# ============================================================ helpers
@app.cell
def _():
    MAX_DISPLAY = 40  # cap heatmap size so long prompts stay legible (PRD risk §18)

    def clean_token(t: str) -> str:
        """GPT-2 byte-level: 'Ġ' marks a leading space, 'Ċ' a newline."""
        return t.replace("Ġ", "␣").replace("Ċ", "⏎") or "∅"

    def pos_label(i: int, t: str) -> str:
        return f"{i:02d} {clean_token(t)}"

    return MAX_DISPLAY, clean_token, pos_label


# ============================================================ presets + committed state
@app.cell
def _(mo):
    PRESETS = {
        "A normal sentence": "The quick brown fox jumps over the lazy dog.",
        "A repeated phrase": "I really really really really want to go home now.",
        "A very short prompt": "Hello there.",
        "A long prompt": (
            "In the early morning the city slowly wakes, traffic begins to hum "
            "along the avenues, vendors raise their shutters, and commuters spill "
            "out of the stations into the pale gold light of another working day."
        ),
        "Starts with punctuation": "— and then everything changed in an instant.",
        "A repeated prefix": "Paris Paris Paris is the capital of France.",
    }
    DEFAULT_LABEL = next(iter(PRESETS))

    # Single source of truth for the analysed prompt. Updated by the preset
    # dropdown (on_change) and by the Run button — NOT by every keystroke.
    get_prompt, set_prompt = mo.state(PRESETS[DEFAULT_LABEL])
    return DEFAULT_LABEL, PRESETS, get_prompt, set_prompt


# ============================================================ controls
@app.cell
def _(DEFAULT_LABEL, PRESETS, mo, set_prompt):
    preset_dd = mo.ui.dropdown(
        options=PRESETS,
        value=DEFAULT_LABEL,
        on_change=lambda v: set_prompt(v) if v is not None else None,
    )
    prompt_area = mo.ui.text_area(
        placeholder="Type your own prompt, then press Run ▸",
        rows=2,
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Run ▸")
    return preset_dd, prompt_area, run_button


@app.cell
def _(prompt_area, run_button, set_prompt):
    # Commit custom text only on a Run click. Typing reruns this cheap cell but
    # does not touch state, so inference (Cell B) never fires per keystroke.
    if run_button.value and prompt_area.value.strip():
        set_prompt(prompt_area.value.strip())
    return


# ============================================================ Cell B driver
@app.cell
def _(MAX_DISPLAY, get_prompt, run_inference):
    prompt = get_prompt()
    bundle = run_inference(prompt)

    attn = bundle["attn"]              # (L, H, S, S)
    hidden = bundle["hidden"]          # (L+1, S, D)
    tokens = bundle["tokens"]
    n_layers, n_heads, seq_len, _ = attn.shape

    truncated = seq_len > MAX_DISPLAY
    disp = min(seq_len, MAX_DISPLAY)
    return (
        attn,
        disp,
        hidden,
        n_heads,
        n_layers,
        prompt,
        seq_len,
        tokens,
        truncated,
    )


# ============================================================ Cell C — metrics (cheap)
@app.cell
def _(attn, n_layers, seq_len):
    # Layer-wise sink: mean attention to token 0, averaged over heads,
    # EXCLUDING the trivial query row 0 (causal mask makes it self-attend 100%).
    if seq_len > 1:
        layerwise = attn[:, :, 1:, 0].mean(dim=(1, 2))          # (L,)
        overall_sink = attn[:, :, 1:, 0].mean().item()
    else:
        layerwise = attn[:, :, :, 0].mean(dim=(1, 2))
        overall_sink = 1.0
    peak_sink = layerwise.max().item()
    peak_layer = int(layerwise.argmax().item())
    return layerwise, overall_sink, peak_layer, peak_sink


@app.cell
def _(hidden, n_layers, seq_len):
    # Over-mixing meter (PRD stretch §9.1): mean pairwise cosine similarity
    # between token representations at each layer, EXCLUDING the sink token 0, so
    # we measure collapse among the tokens that carry meaning. Higher = more
    # over-mixed. hidden[1:] aligns the L+1 hidden states to the L attn layers.
    collapse = []
    for Lh in range(1, n_layers + 1):
        H = hidden[Lh][1:] if seq_len > 1 else hidden[Lh]
        if H.shape[0] > 1:
            Hn = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sim = Hn @ Hn.T
            n = sim.shape[0]
            off = (sim.sum() - sim.diagonal().sum()) / (n * n - n)
            collapse.append(float(off))
        else:
            collapse.append(float("nan"))
    deepest_collapse = collapse[-1] if collapse else float("nan")
    return collapse, deepest_collapse


# ============================================================ shared layer state
@app.cell
def _(mo, peak_layer):
    # One source of truth for the displayed layer, written by BOTH the slider and
    # the autoplay ticker — so "▶ play depth" can sweep every depth-linked view
    # (heatmap, sink curve, collapse scatter) at once. Re-runs only when a new
    # prompt is analysed, which resets the view to that prompt's peak layer.
    get_layer, set_layer = mo.state(peak_layer)
    return get_layer, set_layer


# ============================================================ layer / head selectors
@app.cell
def _(get_layer, mo, n_heads, n_layers, set_layer):
    layer_sel = mo.ui.slider(
        0, n_layers - 1, value=get_layer(), step=1, label="layer",
        show_value=True, on_change=set_layer,
    )
    head_sel = mo.ui.slider(
        0, n_heads - 1, value=0, step=1, label="head", show_value=True
    )
    playing = mo.ui.checkbox(label="▶ play depth")
    ticker = mo.ui.refresh(options=["0.5s", "0.8s", "1.2s"], default_interval="0.8s")
    return head_sel, layer_sel, playing, ticker


# ============================================================ autoplay ticker
@app.cell
def _(n_layers, playing, set_layer, ticker):
    # Each tick advances the shared layer (wrapping at the top) while "play" is on.
    # A functional update means this cell never *reads* the layer state, so it fires
    # on the ticker alone and can't retrigger itself into a feedback loop.
    ticker.value
    if playing.value:
        set_layer(lambda cur: (cur + 1) % n_layers)
    return


# ============================================================ per-selection metrics
@app.cell
def _(attn, head_sel, layer_sel, seq_len):
    L, Hd = layer_sel.value, head_sel.value
    A = attn[L, Hd]  # (S, S): row = query, col = key

    # Headline sink for the current head, and averaged over all heads at layer L.
    sink_head = A[1:, 0].mean().item() if seq_len > 1 else 1.0
    sink_layer_allheads = (
        attn[L, :, 1:, 0].mean().item() if seq_len > 1 else 1.0
    )

    # Per-token received attention: masked column mean over valid (causal) query
    # rows, skipping the trivial row 0.
    received = []
    for j in range(seq_len):
        col = A[max(j, 1):, j]
        received.append(col.mean().item() if col.numel() > 0 else A[j, j].item())
    return A, Hd, L, received, sink_head, sink_layer_allheads


# ============================================================================
#  NARRATIVE  +  VISUALS  (top-to-bottom reading order, PRD §10 / §16)
# ============================================================================


# ------------------------------------------------------------ hero
@app.cell
def _(SERIF, mo, n_heads, n_layers, overall_sink, peak_sink):
    s_avg = int(round(overall_sink * 100))
    s_peak = int(round(peak_sink * 100))
    # The two % stats count up from 0 via animated @property custom properties.
    # Fallbacks are layered: if @property is unsupported the counter rests on its
    # initial-value (the true number); if the whole <style> is stripped, the plain
    # text inside the span shows instead. Either way a correct number is visible.
    mo.md(
        f"""
<style>
@property --aspA {{ syntax:'<integer>'; initial-value:{s_avg}; inherits:true; }}
@property --aspB {{ syntax:'<integer>'; initial-value:{s_peak}; inherits:true; }}
@keyframes aspCountA {{ from {{ --aspA:0 }} to {{ --aspA:{s_avg} }} }}
@keyframes aspCountB {{ from {{ --aspB:0 }} to {{ --aspB:{s_peak} }} }}
@keyframes aspRise {{ from {{ opacity:0; transform:translateY(9px) }}
                      to {{ opacity:1; transform:none }} }}
@keyframes aspFill {{ from {{ width:0 }} to {{ width:var(--w) }} }}
.asp-stats {{ display:flex; gap:24px; margin-top:10px; flex-wrap:wrap; }}
.asp-stat {{ animation:aspRise .7s ease-out both; }}
.asp-stat:nth-child(2) {{ animation-delay:.09s; }}
.asp-stat:nth-child(3) {{ animation-delay:.18s; }}
.asp-num {{ font:600 30px {SERIF}; line-height:1; }}
.asp-a, .asp-b {{ font-size:0 !important; animation-duration:1.1s;
  animation-timing-function:ease-out; animation-fill-mode:both; }}
.asp-a {{ animation-name:aspCountA; }}
.asp-b {{ animation-name:aspCountB; }}
.asp-a::after {{ font:600 30px {SERIF}; color:#2E6BDB;
  counter-reset:aspA var(--aspA); content:counter(aspA) '%'; }}
.asp-b::after {{ font:600 30px {SERIF}; color:#14181F;
  counter-reset:aspB var(--aspB); content:counter(aspB) '%'; }}
.asp-bar {{ height:3px; width:74px; border-radius:2px; background:#E2E8F0;
  margin-top:6px; overflow:hidden; }}
.asp-bar > i {{ display:block; height:100%; background:#2E6BDB;
  animation:aspFill 1.1s ease-out both; }}
.asp-cap {{ font:500 10px monospace; color:#7A828D; display:block; margin-top:4px; }}
</style>

<div style="font:600 11px {('IBM Plex Mono, monospace')};letter-spacing:.16em;
text-transform:uppercase;color:#2E6BDB;">An interactive marimo notebook ·
arXiv:2504.02732 · COLM 2025</div>

<div style="font:500 44px/1.08 {SERIF};color:#14181F;letter-spacing:-.015em;
margin:.3em 0;">Why do language models pour attention onto their
<em style="color:#2E6BDB;">first token</em>?</div>

Every LLM has a strange habit: it dumps a huge share of attention onto a token
that often means nothing. Type a sentence, watch it happen live, then **break it
on purpose** — and learn why it's a feature, not a bug.

<div class="asp-stats">
  <div class="asp-stat">
    <span class="asp-num asp-a" style="color:#2E6BDB;">{s_avg}%</span>
    <div class="asp-bar"><i style="--w:{s_avg}%;"></i></div>
    <span class="asp-cap">avg attention → token 0</span>
  </div>
  <div class="asp-stat">
    <span class="asp-num asp-b" style="color:#14181F;">{s_peak}%</span>
    <div class="asp-bar"><i style="--w:{s_peak}%;background:#14181F;"></i></div>
    <span class="asp-cap">peak across layers</span>
  </div>
  <div class="asp-stat">
    <span class="asp-num" style="color:#14181F;">{n_layers}×{n_heads}</span>
    <span class="asp-cap">layers × heads · gpt2</span>
  </div>
</div>
"""
    )
    return


# ------------------------------------------------------------ controls panel
@app.cell
def _(mo, preset_dd, prompt_area, run_button):
    mo.vstack(
        [
            mo.md("**Pick a preset** — or type your own and press Run."),
            mo.hstack(
                [preset_dd, run_button], justify="start", align="center", gap=1
            ),
            prompt_area,
        ],
        gap=0.6,
    )
    return


# ------------------------------------------------------------ 01 · tokens
@app.cell
def _(ACCENT, BORDER, clean_token, disp, mo, tokens, truncated):
    import html

    chips = []
    for i, t in enumerate(tokens[:disp]):
        first = i == 0
        # Escape: token text is user-derived and rendered as raw HTML here.
        label = html.escape(clean_token(t))
        chips.append(
            f"""<span style="display:inline-block;margin:2px;padding:4px 9px;
border-radius:6px;font:500 13px 'IBM Plex Mono',monospace;
border:1px solid {ACCENT if first else BORDER};
background:{ACCENT if first else '#FFFFFF'};
color:{'#FFFFFF' if first else '#14181F'};">{label}</span>"""
        )
    note = (
        f"<div style='font:500 11px monospace;color:#94A0AD;margin-top:6px;'>"
        f"showing first {disp} tokens (prompt is longer)</div>"
        if truncated
        else ""
    )
    mo.md(
        f"""### 01 · Tokens — first, what is a token?

A model never sees letters or words. It chops text into **tokens** — sometimes a
whole word, sometimes a piece of one, sometimes punctuation. The first token
(filled blue) is the one everything will end up pointing at.

<div style="line-height:2.1;">{''.join(chips)}</div>{note}

> **The first token is not the first letter.** `␣` marks a leading space in
> GPT-2's byte-level tokenizer. GPT-2 adds no special start token, so the "first
> token" here is simply the first real chunk of your prompt.
"""
    )
    return


# ------------------------------------------------------------ 01b · tokenization in 3-D (three.js)
@app.cell
def _(ACCENT, clean_token, tokens):
    import json as _json

    tokens3d_json = _json.dumps(
        {
            "accent": ACCENT,
            "tokens": [
                {"text": clean_token(t), "first": i == 0} for i, t in enumerate(tokens)
            ],
        }
    )
    return (tokens3d_json,)


@app.cell
def _(mo, tokens3d_json):
    # Same verified three.js + same-origin srcdoc pattern as §06b. Each token is a
    # labelled 3-D tile (canvas-texture, so no font files); they start packed edge-
    # to-edge as the raw sentence, then animate apart into the chunks the model sees.
    _tpl = """<!doctype html><html><head><meta charset="utf-8"/>
<style>
  html,body{margin:0;background:#F8FAFC;font-family:ui-sans-serif,sans-serif;}
  #wrap{position:relative;width:100%;height:420px;}
  #cv{width:100%;height:100%;display:block;touch-action:none;cursor:grab;}
  #cv:active{cursor:grabbing;}
  #ctrls{position:absolute;left:14px;bottom:14px;right:14px;display:flex;align-items:center;
    gap:12px;background:rgba(255,255,255,.82);backdrop-filter:blur(6px);border:1px solid #E2E8F0;
    border-radius:10px;padding:8px 12px;font:500 12px 'IBM Plex Mono',ui-monospace,monospace;color:#4E5663;}
  #replay{border:1px solid #2E6BDB;background:#2E6BDB;color:#fff;border-radius:7px;padding:5px 12px;cursor:pointer;font:inherit;}
  .lab{white-space:nowrap;}
  #err{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:24px;
    color:#7A828D;font-size:13px;background:#F8FAFC;}
</style></head><body>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="ctrls">
    <button id="replay">&#8635; re-cut</button>
    <span class="lab"><b id="ntok">0</b> tokens</span>
    <span class="lab" style="color:#7A828D;flex:1;">one bar of text &rarr; the chunks the model sees &middot; drag to orbit</span>
  </div>
  <div id="err">Rendering the tokens&hellip;</div>
</div>
<script type="importmap">
{ "imports": { "three": "https://esm.sh/three@0.160.0", "three/addons/": "https://esm.sh/three@0.160.0/examples/jsm/" } }
</script>
<script type="module">
const DATA = __DATA__;
const ERR = document.getElementById("err");
try {
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const wrap=document.getElementById("wrap"), canvas=document.getElementById("cv");
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(42,1,0.01,200);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=0.08; controls.enablePan=false;
  scene.add(new THREE.AmbientLight(0xffffff,1));
  function tileTexture(text, bg, fg){
    const fontPx=72, pad=28, c=document.createElement("canvas"), x=c.getContext("2d");
    x.font=`600 ${fontPx}px 'IBM Plex Mono', monospace`;
    const w=Math.ceil(x.measureText(text).width)+pad*2, h=fontPx+pad*2;
    c.width=w; c.height=h;
    x.fillStyle=bg; x.fillRect(0,0,w,h);
    x.font=`600 ${fontPx}px 'IBM Plex Mono', monospace`;
    x.fillStyle=fg; x.textBaseline="middle"; x.textAlign="center";
    x.fillText(text,w/2,h/2+2);
    const t=new THREE.CanvasTexture(c); t.anisotropy=4; t.needsUpdate=true;
    return {tex:t, aspect:w/h};
  }
  const H=1.0, GAP=0.5, DEPTH=0.14;
  const tiles=[], widths=[];
  DATA.tokens.forEach(tk=>{
    const bg = tk.first ? "#2E6BDB" : "#FFFFFF";
    const fg = tk.first ? "#FFFFFF" : "#14181F";
    const {tex,aspect}=tileTexture(tk.text||"∅", bg, fg);
    const w=Math.max(0.5, aspect*H);
    const side=new THREE.MeshBasicMaterial({color: tk.first ? 0x2E6BDB : 0xE7ECF3});
    const face=new THREE.MeshBasicMaterial({map:tex});
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,H,DEPTH),
      [side,side,side,side,face,face]);
    scene.add(mesh); tiles.push(mesh); widths.push(w);
  });
  const n=tiles.length;
  function centers(gap){
    const total=widths.reduce((a,b)=>a+b,0)+gap*(n-1);
    let x=-total/2, out=[];
    for(let i=0;i<n;i++){ out.push(x+widths[i]/2); x+=widths[i]+gap; }
    return out;
  }
  const packed=centers(0.01), spread=centers(GAP);          // solid bar vs separated chunks
  const rowW=(spread[n-1]+widths[n-1]/2)-(spread[0]-widths[0]/2);
  let startT=performance.now(), fitted=false;
  const DUR=1500, STAG=0.55;                                // ~1.5s unzip, left-to-right
  const ease=u=>u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
  function resize(){
    const w=wrap.clientWidth,h=wrap.clientHeight;
    renderer.setSize(w,h,false);
    camera.aspect=w/h; camera.updateProjectionMatrix();
    if(!fitted){                                            // frame the whole sentence once
      const vF=camera.fov*Math.PI/180, hF=2*Math.atan(Math.tan(vF/2)*camera.aspect);
      const z=Math.max((rowW/2+0.4)/Math.tan(hF/2),(H*1.6/2)/Math.tan(vF/2))+0.8;
      camera.position.set(0,0.7,z);
      controls.minDistance=z*0.5; controls.maxDistance=z*2.4; controls.target.set(0,0,0);
      controls.update(); fitted=true;
    }
  }
  window.addEventListener("resize",resize); resize();
  function loop(){
    requestAnimationFrame(loop);
    // Time-based so the cut always reads at ~1.5s, whatever the frame rate.
    const t = Math.min(1, (performance.now()-startT)/DUR);
    for(let i=0;i<n;i++){
      const p = Math.min(1, Math.max(0, t*(1+STAG) - (n>1 ? i/(n-1) : 0)*STAG));
      const e = ease(p);
      tiles[i].position.x = packed[i] + (spread[i]-packed[i])*e;
      tiles[i].position.z = Math.sin(e*Math.PI) * 0.45 * ((i%2)?1:-1);
      tiles[i].position.y = Math.sin(e*Math.PI) * 0.12;
    }
    controls.update();
    renderer.render(scene,camera);
  }
  loop();
  document.getElementById("ntok").textContent=String(n);
  document.getElementById("replay").addEventListener("click",()=>{ startT=performance.now(); });
  ERR.style.display="none";
} catch (e) {
  ERR.textContent="3-D view couldn't load here — the token chips above show the same split.";
  ERR.style.display="grid";
  console.error("token3d init failed", e);
}
</script></body></html>"""
    import html as _html

    _doc = _tpl.replace("__DATA__", tokens3d_json)
    _frame = (
        '<iframe sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:440px;border:1px solid #E2E8F0;border-radius:12px;" '
        'srcdoc="' + _html.escape(_doc, quote=True) + '"></iframe>'
    )
    mo.vstack(
        [
            mo.md(
                "**See the split in 3-D.** Your sentence starts as one solid bar, then "
                "**watch it break into the chunks the model actually reads** — drag to "
                "orbit, and hit **↻ replay** to re-cut. Notice the `␣` on most tiles: the "
                "leading space is *part of* the token, which is why the "
                "<span style='color:#2E6BDB;font-weight:600;'>first token</span> isn't "
                "simply the first letter."
            ),
            mo.Html(_frame),
        ],
        gap=0.6,
    )
    return


# ------------------------------------------------------------ 02 · attention heatmap
@app.cell
def _(A, disp, mo, pd, pos_label, styled, tokens):
    import altair as _alt

    rows = []
    labels = [pos_label(i, tokens[i]) for i in range(disp)]
    for qi in range(disp):
        for ki in range(disp):
            rows.append(
                {
                    "query": labels[qi],
                    "key": labels[ki],
                    "weight": float(A[qi, ki]),
                }
            )
    df = pd.DataFrame(rows)

    heat = (
        _alt.Chart(df)
        .mark_rect()
        .encode(
            x=_alt.X("key:O", sort=labels, title="key  (attended TO)"),
            y=_alt.Y("query:O", sort=labels, title="query  (attending FROM)"),
            color=_alt.Color(
                "weight:Q",
                scale=_alt.Scale(scheme="blues"),
                legend=_alt.Legend(title="attention"),
            ),
            tooltip=[
                _alt.Tooltip("query:N"),
                _alt.Tooltip("key:N"),
                _alt.Tooltip("weight:Q", format=".1%"),
            ],
        )
    )
    mo.md(
        f"""### 02 · Attention — who listens to whom?

Each token decides how much to **listen to** each earlier token. Read a row
left-to-right: that's where one token sends its attention. Watch the leftmost
column (token 0) — it tends to glow.
"""
    )
    return df, heat, labels


@app.cell
def _(heat, mo, styled):
    mo.ui.altair_chart(styled(heat, h=460))
    return


# ------------------------------------------------------------ 03 · the sink
@app.cell
def _(
    ACCENT,
    L,
    SERIF,
    disp,
    labels,
    mo,
    pd,
    received,
    sink_head,
    sink_layer_allheads,
    styled,
    tokens,
):
    import altair as _alt

    bar_df = pd.DataFrame(
        {"token": labels, "received": [received[i] for i in range(disp)]}
    )
    bar_df["is_first"] = [i == 0 for i in range(disp)]

    bars = (
        _alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=_alt.X("token:O", sort=labels, title=None, axis=_alt.Axis(labelAngle=-45)),
            y=_alt.Y("received:Q", title="attention received", axis=_alt.Axis(format="%")),
            color=_alt.condition(
                _alt.datum.is_first, _alt.value(ACCENT), _alt.value("#C9D6E8")
            ),
            tooltip=[_alt.Tooltip("token:N"), _alt.Tooltip("received:Q", format=".1%")],
        )
    )
    ann = (
        _alt.Chart(pd.DataFrame({"token": [labels[0]], "received": [received[0]]}))
        .mark_text(text="the sink ▲", dy=-9, color=ACCENT, fontSize=11, fontWeight=600)
        .encode(x=_alt.X("token:O", sort=labels), y=_alt.Y("received:Q"))
    )

    stat = mo.md(
        f"""<div style="display:flex;gap:24px;align-items:baseline;">
<div style="font:600 46px {SERIF};color:#2E6BDB;">{sink_head:.0%}</div>
<div style="font:400 15px sans-serif;color:#4E5663;">of attention from every other
token lands on <b>token 0</b>, at this head.<br>
<span style="font:500 12px monospace;color:#7A828D;">
all heads at layer {L}: {sink_layer_allheads:.0%}</span></div></div>"""
    )

    mo.vstack(
        [
            mo.md(
                "### 03 · There's a sink\n\nTotal up the attention each token "
                "*receives* and one bar towers over the rest. That spike on the "
                "first token is an **attention sink**."
            ),
            stat,
            mo.ui.altair_chart(styled(bars + ann, h=340)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 3b · verdict gauge
@app.cell
def _(SERIF, mo, overall_sink):
    v = overall_sink
    if v < 0.30:
        word, blurb = "a mild sink", "only a little spare attention parks on token 0"
    elif v < 0.55:
        word, blurb = "a moderate sink", "a good share of attention parks on token 0"
    elif v < 0.75:
        word, blurb = "a strong sink", "most spare attention parks on the first token"
    else:
        word, blurb = "a very strong sink", "the first token is doing heavy lifting"
    pct = int(round(v * 100))
    mo.md(
        f"""
<style>
@keyframes aspGauge {{ from {{ width:0 }} to {{ width:{pct}% }} }}
.asp-g-track {{ height:10px; border-radius:6px;
  background:linear-gradient(90deg,#EEF2F7,#DCE6F5); overflow:hidden; }}
.asp-g-fill {{ height:100%; border-radius:6px;
  background:linear-gradient(90deg,#7FA8E8,#2E6BDB); animation:aspGauge 1s ease-out both; }}
.asp-g-row {{ display:flex; justify-content:space-between; font:500 10px monospace;
  color:#94A0AD; margin-top:4px; }}
</style>

<div style="border:1px solid #E2E8F0;border-radius:12px;padding:14px 16px;
background:#FFFFFF;margin:.4em 0;">
<div style="font:600 16px {SERIF};color:#14181F;margin-bottom:9px;">
Your sentence has <span style="color:#2E6BDB;">{word}</span> — {pct}% of spare
attention pools on token&nbsp;0.</div>
<div class="asp-g-track"><div class="asp-g-fill" style="width:{pct}%;"></div></div>
<div class="asp-g-row"><span>0%</span><span>{blurb}</span><span>100%</span></div>
</div>
"""
    )
    return


# ------------------------------------------------------------ 04 · why
@app.cell
def _(mo):
    mo.md(
        """### 04 · Why would a model *do* this?

Imagine a classroom where everyone keeps copying a little from their neighbours'
notes. A bit helps — but copy round after round and every page turns into the
same blurry soup. Nobody keeps an original thought. Researchers call that
**over-mixing** (a.k.a. rank / representational collapse).

A Transformer mixes tokens the same way, layer after layer. The paper's answer:
the model learns to **dump spare attention on the first token** — a near-empty
parking spot — instead of forcing more mixing between tokens that actually
matter. The sink is a *release valve* that keeps representations distinct. Not a
bug; a learned defense.
"""
    )
    return


# ------------------------------------------------------------ 05 · grows with depth
@app.cell
def _(head_sel, layer_sel, layerwise, mo, n_layers, pd, playing, styled, ticker):
    import altair as _alt

    curve_df = pd.DataFrame(
        {"layer": list(range(n_layers)), "sink": [float(v) for v in layerwise]}
    )
    line = (
        _alt.Chart(curve_df)
        .mark_line(point=True, color="#2E6BDB", strokeWidth=2.5)
        .encode(
            x=_alt.X("layer:Q", title="layer (depth →)", axis=_alt.Axis(tickMinStep=1)),
            y=_alt.Y("sink:Q", title="avg attention → token 0", axis=_alt.Axis(format="%")),
            tooltip=[_alt.Tooltip("layer:Q"), _alt.Tooltip("sink:Q", format=".1%")],
        )
    )
    peak_txt = (
        _alt.Chart(pd.DataFrame({"layer": [n_layers - 1], "sink": [float(layerwise.max())]}))
        .mark_text(text="stronger with depth →", align="right", dx=-4, dy=-8,
                   color="#7A828D", fontSize=10)
        .encode(x="layer:Q", y="sink:Q")
    )
    rule = (
        _alt.Chart(pd.DataFrame({"layer": [layer_sel.value]}))
        .mark_rule(color="#D9634F", strokeDash=[4, 3])
        .encode(x="layer:Q")
    )

    mo.vstack(
        [
            mo.md(
                "### 05 · It grows with depth\n\nDrag the **layer** slider — or hit "
                "**▶ play depth** — and watch the heatmap, this curve, and the "
                "collapse plot just below all move together. Averaged over heads, "
                "the sink generally *strengthens* in deeper layers — exactly what "
                "the paper predicts: deeper networks need stronger sinks to avoid "
                "collapse."
            ),
            mo.hstack(
                [layer_sel, head_sel, playing, ticker],
                justify="start", gap=1.5, align="center",
            ),
            mo.ui.altair_chart(styled(line + rule + peak_txt, h=340)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 06 · collapse scatter (2-D PCA)
@app.cell
def _(hidden, n_layers, np, pd, seq_len):
    # Make representational collapse VISIBLE. Two deliberate choices so the picture
    # matches the over-mixing meter (which measures cosine similarity):
    #   1. Unit-normalize each token -> the projection captures DIRECTION, not norm.
    #      Deep layers have huge-norm vectors; without this they spread by magnitude
    #      and the dots drift *apart* with depth, contradicting the story.
    #   2. Center each layer's content cloud at the origin -> we see the cloud's
    #      *size* (the collapse) instead of its drift across the frame, so the dots
    #      genuinely pull together as cosine similarity climbs.
    # Locals are underscore-prefixed so marimo keeps them cell-private.
    _H = hidden.numpy()                          # (L+1, S, D)
    _H = _H / np.linalg.norm(_H, axis=-1, keepdims=True).clip(1e-8)
    _, _S, _D = _H.shape
    _fit = _H[:, 1:, :].reshape(-1, _D) if seq_len > 1 else _H.reshape(-1, _D)
    _mean = _fit.mean(axis=0)
    _, _, _Vt = np.linalg.svd(_fit - _mean, full_matrices=False)
    _proj = (_H - _mean) @ _Vt[:2].T             # (L+1, S, 2)
    _rows = []
    for _Li in range(n_layers):                  # hidden[_Li+1] = output of attn layer _Li
        _P = _proj[_Li + 1]
        _c = _P[1:].mean(axis=0) if _S > 1 else _P[0]   # content centroid
        for _ti in range(_S):
            _xy = _P[_ti] - _c
            _rows.append(
                {
                    "layer": _Li,
                    "tok": _ti,
                    "x": float(_xy[0]),
                    "y": float(_xy[1]),
                    "kind": "first token" if _ti == 0 else "content token",
                }
            )
    scatter_all = pd.DataFrame(_rows)
    return (scatter_all,)


@app.cell
def _(ACCENT, L, MONO, MUTED, clean_token, collapse, mo, scatter_all, styled, tokens):
    import altair as _alt

    _pts_df = scatter_all[scatter_all["layer"] == L].copy()
    _pts_df["token"] = [
        tokens[int(t)] if int(t) < len(tokens) else str(t) for t in _pts_df["tok"]
    ]
    _pts_df["label"] = [clean_token(t) for t in _pts_df["token"]]
    # Frame = the content cloud's widest extent across ALL layers (symmetric about
    # the origin, since each layer is centered), padded. Fixed across layers, so a
    # shrinking cloud reads as collapse — not a silent rescale. The first token can
    # land outside this frame at shallow layers; clamp it to the edge so it stays
    # visible (and visibly apart) without blowing out the view.
    _content = scatter_all[scatter_all["kind"] == "content token"]
    _src = _content if len(_content) else scatter_all
    _rx = float(_src["x"].abs().max()) * 1.18 or 1.0
    _ry = float(_src["y"].abs().max()) * 1.18 or 1.0
    _xr = [-_rx, _rx]
    _yr = [-_ry, _ry]
    _pts_df["x"] = _pts_df["x"].clip(-_rx, _rx)
    _pts_df["y"] = _pts_df["y"].clip(-_ry, _ry)

    # Labelled (but tick-free) PCA axes + a faint grid give the eye a reference
    # frame; the raw PCA numbers are arbitrary, so we hide them. mark_text prints
    # each token next to its dot, so it's concrete (you see the words) and the
    # overlap when they clump *is* the collapse.
    _ax_x = _alt.Axis(title="principal component 1", labels=False, ticks=False,
                      grid=True, gridColor="#EEF2F7", domainColor="#E2E8F0")
    _ax_y = _alt.Axis(title="principal component 2", labels=False, ticks=False,
                      grid=True, gridColor="#EEF2F7", domainColor="#E2E8F0")
    _color = _alt.Color(
        "kind:N",
        scale=_alt.Scale(domain=["content token", "first token"], range=["#9DB3D6", ACCENT]),
        legend=_alt.Legend(title=None, orient="top"),
    )
    _pts = (
        _alt.Chart(_pts_df)
        .mark_circle(size=300, opacity=0.9, stroke="#FFFFFF", strokeWidth=1)
        .encode(
            x=_alt.X("x:Q", scale=_alt.Scale(domain=_xr), axis=_ax_x),
            y=_alt.Y("y:Q", scale=_alt.Scale(domain=_yr), axis=_ax_y),
            color=_color,
            tooltip=[_alt.Tooltip("token:N"), _alt.Tooltip("kind:N")],
        )
    )
    _labels = (
        _alt.Chart(_pts_df)
        .mark_text(fontSize=10, dy=-13, font=MONO, color=MUTED)
        .encode(
            x=_alt.X("x:Q", scale=_alt.Scale(domain=_xr)),
            y=_alt.Y("y:Q", scale=_alt.Scale(domain=_yr)),
            text="label:N",
        )
    )
    _sim = collapse[L] if L < len(collapse) else float("nan")
    _sim_txt = f"{_sim:.0%}" if _sim == _sim else "—"   # nan-safe (single-token prompts)

    mo.vstack(
        [
            mo.md(
                f"""### 06 · Watch the tokens collapse

Each dot is one token's **internal representation** — *direction only*
(unit-normalized), flattened to 2-D. Drag the **layer** slider above, or hit
**▶ play depth**, and watch the content tokens
(<span style="color:#9DB3D6;font-weight:600;">pale</span>) **pull together into a
single clump** as you go deeper: by this layer they're **{_sim_txt} similar** to
one another — exactly the *over-mixing* the paper warns about. The
<span style="color:{ACCENT};font-weight:600;">first token</span> is highlighted
for reference (it sits apart in the early layers); the sink is what keeps the
meaningful tokens from collapsing even faster.

_Direction-only (cosine) 2-D PCA, each layer centered on a shared frame —
illustrative, in the spirit of the paper, not its formal rank measure._"""
            ),
            mo.ui.altair_chart(styled(_pts + _labels, h=460, frame=True)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 6b · 3-D collapse (three.js)
@app.cell
def _(ACCENT, hidden, n_layers, np, seq_len):
    import json as _json

    # Same recipe as the 2-D scatter (unit-normalize -> direction; center each
    # layer on its content centroid) but keep THREE principal components for an
    # orbitable cloud. Pre-scale into a [-1,1] cube so the JS just plots points.
    _H = hidden.numpy()
    _H = _H / np.linalg.norm(_H, axis=-1, keepdims=True).clip(1e-8)
    _, _S, _D = _H.shape
    _fit = _H[:, 1:, :].reshape(-1, _D) if seq_len > 1 else _H.reshape(-1, _D)
    _mean = _fit.mean(axis=0)
    _, _, _Vt = np.linalg.svd(_fit - _mean, full_matrices=False)
    _proj = (_H - _mean) @ _Vt[:3].T              # (L+1, S, 3)
    _cl = []
    for _Li in range(n_layers):
        _P = _proj[_Li + 1]
        _c = _P[1:].mean(axis=0) if _S > 1 else _P[0]
        _cl.append(_P - _c)
    _cl = np.stack(_cl)                            # (L, S, 3), per-layer centered
    _content = _cl[:, 1:, :] if _S > 1 else _cl
    _scale = np.abs(_content).max(axis=(0, 1)).clip(1e-6)   # per-axis (3,)
    _norm = np.clip(_cl / _scale * 0.9, -1.0, 1.0)
    viz3d_json = _json.dumps(
        {
            "accent": ACCENT,
            "layers": int(n_layers),
            "points": [
                [
                    {
                        "x": round(float(_norm[_Li, _ti, 0]), 4),
                        "y": round(float(_norm[_Li, _ti, 1]), 4),
                        "z": round(float(_norm[_Li, _ti, 2]), 4),
                        "first": _ti == 0,
                    }
                    for _ti in range(_S)
                ]
                for _Li in range(n_layers)
            ],
        }
    )
    return (viz3d_json,)


@app.cell
def _(mo, viz3d_json):
    # Self-contained three.js widget (sandboxed iframe). It owns its OWN layer
    # slider + play button so orbiting/animating never round-trips to Python.
    # If three.js can't load, the #err overlay shows and the 2-D scatter above
    # still carries the story — so this is purely additive.
    _tpl = """<!doctype html><html><head><meta charset="utf-8"/>
<style>
  html,body{margin:0;background:#F8FAFC;font-family:ui-sans-serif,sans-serif;}
  #wrap{position:relative;width:100%;height:520px;}
  #cv{width:100%;height:100%;display:block;touch-action:none;cursor:grab;}
  #cv:active{cursor:grabbing;}
  #ctrls{position:absolute;left:14px;bottom:14px;right:14px;display:flex;align-items:center;
    gap:12px;background:rgba(255,255,255,.82);backdrop-filter:blur(6px);border:1px solid #E2E8F0;
    border-radius:10px;padding:8px 12px;font:500 12px 'IBM Plex Mono',ui-monospace,monospace;color:#4E5663;}
  #play{border:1px solid #2E6BDB;background:#2E6BDB;color:#fff;border-radius:7px;padding:5px 12px;cursor:pointer;font:inherit;}
  #play.paused{background:#fff;color:#2E6BDB;}
  #range{flex:1;accent-color:#2E6BDB;}
  #err{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:24px;
    color:#7A828D;font-size:13px;background:#F8FAFC;}
  .lab{white-space:nowrap;}
</style></head><body>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="ctrls">
    <button id="play" class="paused">&#9654; play depth</button>
    <span class="lab">layer <b id="lnum">0</b></span>
    <input id="range" type="range" min="0" max="11" value="0" step="1"/>
    <span class="lab" style="color:#7A828D;">drag to orbit &middot; scroll to zoom</span>
  </div>
  <div id="err">Rendering the 3-D cloud&hellip;</div>
</div>
<script type="importmap">
{ "imports": { "three": "https://esm.sh/three@0.160.0", "three/addons/": "https://esm.sh/three@0.160.0/examples/jsm/" } }
</script>
<script type="module">
const DATA = __DATA__;
const ERR = document.getElementById("err");
try {
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  const wrap=document.getElementById("wrap"), canvas=document.getElementById("cv");
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(45,1,0.01,100);
  camera.position.set(1.9,1.4,2.3);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=0.08;
  controls.enableZoom=true; controls.enableRotate=true; controls.enablePan=false;
  controls.minDistance=1.6; controls.maxDistance=7; controls.zoomSpeed=0.8;
  controls.target.set(0,0,0);
  const box=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2,2,2)),
    new THREE.LineBasicMaterial({color:0xdce3ee}));
  scene.add(box);
  const ACCENT=new THREE.Color(DATA.accent), PALE=new THREE.Color("#9DB3D6");
  const n=DATA.points[0].length, spheres=[], cur=[];
  for(let i=0;i<n;i++){
    const first=DATA.points[0][i].first;
    const m=new THREE.Mesh(new THREE.SphereGeometry(first?0.055:0.038,18,18),
      new THREE.MeshBasicMaterial({color:first?ACCENT:PALE}));
    scene.add(m); spheres.push(m);
    const p=DATA.points[0][i]; cur.push(new THREE.Vector3(p.x,p.y,p.z));
  }
  let target=0; const tmp=new THREE.Vector3();
  function resize(){const w=wrap.clientWidth,h=wrap.clientHeight;renderer.setSize(w,h,false);
    camera.aspect=w/h;camera.updateProjectionMatrix();}
  window.addEventListener("resize",resize);
  resize();
  // Canonical continuous render loop: every frame eases points toward the target
  // layer AND calls controls.update() so drag-orbit / scroll-zoom stay responsive.
  function loop(){
    requestAnimationFrame(loop);
    const tp=DATA.points[target];
    for(let i=0;i<n;i++){tmp.set(tp[i].x,tp[i].y,tp[i].z);cur[i].lerp(tmp,0.18);spheres[i].position.copy(cur[i]);}
    controls.update();
    renderer.render(scene,camera);
  }
  loop();
  const range=document.getElementById("range"),lnum=document.getElementById("lnum"),play=document.getElementById("play");
  range.max=String(DATA.layers-1);
  function setLayer(L){range.value=String(L);lnum.textContent=String(L);target=L;}
  range.addEventListener("input",()=>setLayer(+range.value));
  let timer=null;
  play.addEventListener("click",()=>{
    if(timer){clearInterval(timer);timer=null;play.classList.add("paused");play.innerHTML="&#9654; play depth";}
    else{play.classList.remove("paused");play.textContent="❚❚ pause";
      timer=setInterval(()=>setLayer((target+1)%DATA.layers),750);}
  });
  ERR.style.display="none";   // success -> reveal the canvas
} catch (e) {
  ERR.textContent="3-D view couldn't load here — the 2-D chart above tells the same story.";
  ERR.style.display="grid";
  console.error("threejs init failed", e);
}
</script></body></html>"""
    import html as _html

    _doc = _tpl.replace("__DATA__", viz3d_json)
    # Embed via an explicit same-origin srcdoc iframe (mo.Html) rather than
    # mo.iframe: molab's CSP frame-src blocks mo.iframe's blob: URL (renders a
    # broken-image box), but a srcdoc frame is same-origin and allowed. The
    # sandbox keeps it isolated while permitting scripts. If the esm.sh CDN is
    # still blocked, the in-frame #err message shows instead — never a broken box.
    _frame = (
        '<iframe sandbox="allow-scripts allow-same-origin" '
        'style="width:100%;height:560px;border:1px solid #E2E8F0;border-radius:12px;" '
        'srcdoc="' + _html.escape(_doc, quote=True) + '"></iframe>'
    )
    mo.vstack(
        [
            mo.md(
                "**Prefer it in 3-D? Spin the same cloud.** Same direction-only "
                "representations, now with a third principal component — **drag to "
                "orbit**, scroll to zoom, and hit **▶ play depth** to watch the "
                "content tokens collapse inward while the "
                "<span style='color:#2E6BDB;font-weight:600;'>first token</span> "
                "drifts to the edge."
            ),
            mo.Html(_frame),
        ],
        gap=0.6,
    )
    return


# ------------------------------------------------------------ 07 · experiment
@app.cell
def _(mo):
    run_experiment = mo.ui.run_button(label="Run the experiment ▸")
    mo.vstack(
        [
            mo.md(
                "### 07 · Break the first token\n\nThe sink is a *position* effect, "
                "not magic about a special word. Run six prompt variants and compare "
                "how much attention still pools on token 0. Repetition, length, and a "
                "punctuation-first start all move the needle."
            ),
            run_experiment,
        ],
        gap=0.6,
    )
    return (run_experiment,)


@app.cell
def _(ACCENT, mo, overall_sink, pd, prompt, run_experiment, run_inference, seq_len, styled):
    import altair as _alt

    VARIANTS = {
        "normal": "The cat sat on the warm windowsill.",
        "repeated word": "cat cat cat cat cat cat cat cat.",
        "short": "Hi.",
        "long": (
            "The cat sat on the warm windowsill watching the rain fall steadily "
            "over the quiet grey rooftops of the sleeping town all afternoon long."
        ),
        "punct-first": "...the cat sat on the warm windowsill.",
        "repeated prefix": "The The The cat sat on the warm windowsill.",
    }

    mo.stop(
        not run_experiment.value,
        mo.callout(
            mo.md("Press **Run the experiment ▸** to generate your own evidence."),
            kind="info",
        ),
    )

    exp_rows = []
    for name, text in VARIANTS.items():
        b = run_inference(text)
        a = b["attn"]
        s = a.shape[-1]
        sink = a[:, :, 1:, 0].mean().item() if s > 1 else 1.0
        exp_rows.append({"variant": name, "sink": sink, "tokens": s})
    # Drop the live prompt onto the same scale — already inferred, so its sink is
    # free. This is the "leaderboard": see where YOUR sentence ranks.
    exp_rows.append({"variant": "★ your prompt", "sink": overall_sink, "tokens": seq_len})
    exp_df = pd.DataFrame(exp_rows)
    order = list(VARIANTS) + ["★ your prompt"]

    exp_chart = (
        _alt.Chart(exp_df)
        .mark_bar()
        .encode(
            x=_alt.X("variant:N", sort=order, title=None, axis=_alt.Axis(labelAngle=-30)),
            y=_alt.Y("sink:Q", title="avg attention → token 0", axis=_alt.Axis(format="%")),
            color=_alt.condition(
                _alt.datum.variant == "★ your prompt",
                _alt.value("#D9634F"),
                _alt.value(ACCENT),
            ),
            tooltip=[
                _alt.Tooltip("variant:N"),
                _alt.Tooltip("sink:Q", format=".1%"),
                _alt.Tooltip("tokens:Q"),
            ],
        )
    )
    mo.vstack(
        [
            mo.md(
                f"The <span style='color:#D9634F;font-weight:600;'>★ your prompt</span> "
                f"bar is your current sentence (*“{prompt[:48]}{'…' if len(prompt) > 48 else ''}”*) "
                f"— see where it ranks against the six controlled variants."
            ),
            mo.ui.altair_chart(styled(exp_chart, h=360)),
        ],
        gap=0.6,
    )
    return


# ------------------------------------------------------------ 08 · over-mixing meter (stretch)
@app.cell
def _(ACCENT, collapse, deepest_collapse, layerwise, mo, n_layers, pd, styled):
    import altair as _alt

    sink_df = pd.DataFrame(
        {"layer": list(range(n_layers)), "v": [float(x) for x in layerwise]}
    )
    coll_df = pd.DataFrame({"layer": list(range(n_layers)), "v": collapse})

    sink_l = (
        _alt.Chart(sink_df)
        .mark_line(point=True, color=ACCENT, strokeWidth=2.5)
        .encode(
            x=_alt.X("layer:Q", title="layer (depth →)", axis=_alt.Axis(tickMinStep=1)),
            y=_alt.Y(
                "v:Q",
                axis=_alt.Axis(title="attention → token 0", format="%", titleColor=ACCENT),
            ),
            tooltip=[_alt.Tooltip("layer:Q"), _alt.Tooltip("v:Q", format=".1%", title="sink")],
        )
    )
    coll_l = (
        _alt.Chart(coll_df)
        .mark_line(point=True, color="#D9634F", strokeDash=[5, 3], strokeWidth=2.5)
        .encode(
            x=_alt.X("layer:Q"),
            y=_alt.Y(
                "v:Q",
                axis=_alt.Axis(
                    title="token similarity (collapse)", orient="right", titleColor="#D9634F"
                ),
            ),
            tooltip=[
                _alt.Tooltip("layer:Q"),
                _alt.Tooltip("v:Q", format=".2f", title="similarity"),
            ],
        )
    )
    meter = _alt.layer(sink_l, coll_l).resolve_scale(y="independent")

    mo.vstack(
        [
            mo.md(
                f"""### 08 · Feel the collapse  _(stretch)_

The sink is the model's *defense* — here's what it defends against. The
<span style="color:#D9634F;font-weight:600;">red</span> line is how similar the
content tokens' internal representations are to each other at each layer (mean
cosine similarity, token 0 excluded); higher means they're blurring together —
the **over-mixing** the paper warns about. The
<span style="color:#2E6BDB;font-weight:600;">blue</span> line is the sink. They
climb together with depth: as mixing pushes representations toward collapse
(≈{deepest_collapse:.0%} similar by the last layer), the model leans harder on
the first-token parking spot.

_This meter illustrates representational collapse in the spirit of the paper; it
is not a reproduction of its formal measures._"""
            ),
            mo.ui.altair_chart(styled(meter, h=360)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 09 · model-size comparison (stretch)
@app.cell
def _(mo):
    run_compare = mo.ui.run_button(label="Compare model sizes ▸")
    mo.vstack(
        [
            mo.md(
                "### 09 · Scale it up  _(stretch)_\n\nThe paper's most demo-friendly "
                "result: **bigger, deeper models sink harder.** Compare DistilGPT-2 "
                "(6 layers) against GPT-2 (12 layers) on the same prompt, with depth "
                "normalized to 0–1 so they line up. On a molab GPU you can extend the "
                "list to `gpt2-large` and `gpt2-xl` and watch the trend continue — the "
                "GPU does real, motivated work here."
            ),
            run_compare,
        ],
        gap=0.6,
    )
    return (run_compare,)


@app.cell
def _(ACCENT, mo, pd, run_compare, run_inference, styled):
    import altair as _alt

    COMPARE_PROMPT = "The quick brown fox jumps over the lazy dog."
    # CPU-safe defaults (both cached). On a GPU, append "gpt2-large", "gpt2-xl".
    COMPARE_MODELS = ["distilgpt2", "gpt2"]

    mo.stop(
        not run_compare.value,
        mo.callout(
            mo.md("Press **Compare model sizes ▸** to load and overlay the curves."),
            kind="info",
        ),
    )

    comp_rows = []
    for mname in COMPARE_MODELS:
        bd = run_inference(COMPARE_PROMPT, mname)
        ad = bd["attn"]                       # (L, H, S, S)
        Lc = ad.shape[0]
        sd = ad.shape[-1]
        for Li in range(Lc):
            sinkc = ad[Li, :, 1:, 0].mean().item() if sd > 1 else 1.0
            reld = Li / (Lc - 1) if Lc > 1 else 0.0
            comp_rows.append(
                {"model": f"{mname} ({Lc}L)", "rel_depth": reld, "layer": Li, "sink": sinkc}
            )
    comp_df = pd.DataFrame(comp_rows)

    comp_chart = (
        _alt.Chart(comp_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=_alt.X(
                "rel_depth:Q",
                title="relative depth  (0 = first layer · 1 = last)",
                axis=_alt.Axis(format="%"),
            ),
            y=_alt.Y("sink:Q", title="avg attention → token 0", axis=_alt.Axis(format="%")),
            color=_alt.Color(
                "model:N",
                scale=_alt.Scale(range=["#94A0AD", ACCENT]),
                legend=_alt.Legend(title="model (depth)"),
            ),
            tooltip=[
                _alt.Tooltip("model:N"),
                _alt.Tooltip("layer:Q"),
                _alt.Tooltip("sink:Q", format=".1%"),
            ],
        )
    )
    mo.ui.altair_chart(styled(comp_chart, h=380))
    return


# ------------------------------------------------------------ so what
@app.cell
def _(mo):
    mo.md(
        """### So what?

Attention sinks aren't wasted compute or a quirk to patch away — they're a
**learned defense mechanism** against over-mixing, and they get stronger as
models get deeper and contexts get longer. The lesson: even the "boring",
ever-present patterns inside a model can be doing essential work. That's why
interpretability of the mundane matters.

→ Read the paper: **Barbero et al., "Why do LLMs attend to the first token?"**
[arXiv:2504.02732](https://arxiv.org/abs/2504.02732) (COLM 2025).
"""
    )
    return


if __name__ == "__main__":
    app.run()
