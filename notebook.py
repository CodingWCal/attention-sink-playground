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
app = marimo.App(width="medium", app_title="Attention Sink Playground")


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

    def styled(chart, h=260):
        """Apply the shared chart theme to a single-view Altair chart."""
        return (
            chart.properties(height=h, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelFont=MONO,
                titleFont=MONO,
                labelColor=MUTED,
                titleColor=SUBTLE,
                labelFontSize=9,
                titleFontSize=10,
                domainColor=BORDER,
                tickColor=BORDER,
                grid=False,
            )
            .configure_legend(
                labelFont=MONO, titleFont=MONO, labelColor=MUTED, titleColor=SUBTLE
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
        model = AutoModelForCausalLM.from_pretrained(
            name,
            attn_implementation="eager",  # REQUIRED — SDPA/Flash return None attentions
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model = model.to(device).eval()
        return tok, model

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


# ============================================================ layer / head selectors
@app.cell
def _(mo, n_heads, n_layers, peak_layer):
    layer_sel = mo.ui.slider(
        0, n_layers - 1, value=peak_layer, step=1, label="layer", show_value=True
    )
    head_sel = mo.ui.slider(
        0, n_heads - 1, value=0, step=1, label="head", show_value=True
    )
    return head_sel, layer_sel


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


# ------------------------------------------------------------ 1 · hero
@app.cell
def _(SERIF, mo, overall_sink, peak_sink, prompt):
    mo.md(
        f"""
<div style="font:600 11px {('IBM Plex Mono, monospace')};letter-spacing:.16em;
text-transform:uppercase;color:#2E6BDB;">An interactive marimo notebook ·
arXiv:2504.02732 · COLM 2025</div>

<div style="font:500 44px/1.08 {SERIF};color:#14181F;letter-spacing:-.015em;
margin:.3em 0;">Why do language models pour attention onto their
<em style="color:#2E6BDB;">first token</em>?</div>

Every LLM has a strange habit: it dumps a huge share of attention onto a token
that often means nothing. Type a sentence, watch it happen live, then **break it
on purpose** — and learn why it's a feature, not a bug.

<div style="display:flex;gap:14px;margin-top:6px;">
  <div style="font:600 30px {SERIF};color:#2E6BDB;">{overall_sink:.0%}
    <span style="font:500 10px monospace;color:#7A828D;display:block;">
    avg attention → token 0</span></div>
  <div style="font:600 30px {SERIF};color:#14181F;">{peak_sink:.0%}
    <span style="font:500 10px monospace;color:#7A828D;display:block;">
    peak, deepest layers</span></div>
  <div style="font:600 30px {SERIF};color:#14181F;">12×12
    <span style="font:500 10px monospace;color:#7A828D;display:block;">
    layers × heads · gpt2</span></div>
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


# ------------------------------------------------------------ 3 · tokens
@app.cell
def _(ACCENT, BORDER, clean_token, disp, mo, tokens, truncated):
    chips = []
    for i, t in enumerate(tokens[:disp]):
        first = i == 0
        chips.append(
            f"""<span style="display:inline-block;margin:2px;padding:4px 9px;
border-radius:6px;font:500 13px 'IBM Plex Mono',monospace;
border:1px solid {ACCENT if first else BORDER};
background:{ACCENT if first else '#FFFFFF'};
color:{'#FFFFFF' if first else '#14181F'};">{clean_token(t)}</span>"""
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


# ------------------------------------------------------------ 4 · attention heatmap
@app.cell
def _(A, ACCENT, disp, mo, pd, pos_label, styled, tokens):
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
column ({ACCENT and 'token 0'}) — it tends to glow.
"""
    )
    return df, heat, labels


@app.cell
def _(heat, mo, styled):
    mo.ui.altair_chart(styled(heat, h=320))
    return


# ------------------------------------------------------------ 5 · the sink
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
            mo.ui.altair_chart(styled(bars, h=240)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 6 · why
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


# ------------------------------------------------------------ 7 · grows with depth
@app.cell
def _(head_sel, layer_sel, layerwise, mo, n_layers, pd, styled):
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
    rule = (
        _alt.Chart(pd.DataFrame({"layer": [layer_sel.value]}))
        .mark_rule(color="#D9634F", strokeDash=[4, 3])
        .encode(x="layer:Q")
    )

    mo.vstack(
        [
            mo.md(
                "### 05 · It grows with depth\n\nDrag the **layer** slider and watch "
                "the heatmap and the headline number above change. Averaged over "
                "heads, the sink generally *strengthens* in deeper layers — exactly "
                "what the paper predicts: deeper networks need stronger sinks to "
                "avoid collapse."
            ),
            mo.hstack([layer_sel, head_sel], justify="start", gap=2),
            mo.ui.altair_chart(styled(line + rule, h=240)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 8 · experiment
@app.cell
def _(mo):
    run_experiment = mo.ui.run_button(label="Run the experiment ▸")
    mo.vstack(
        [
            mo.md(
                "### 06 · Break the first token\n\nThe sink is a *position* effect, "
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
def _(mo, pd, run_experiment, run_inference, styled):
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
    exp_df = pd.DataFrame(exp_rows)

    exp_chart = (
        _alt.Chart(exp_df)
        .mark_bar(color="#2E6BDB")
        .encode(
            x=_alt.X("variant:N", sort=list(VARIANTS), title=None,
                     axis=_alt.Axis(labelAngle=-30)),
            y=_alt.Y("sink:Q", title="avg attention → token 0", axis=_alt.Axis(format="%")),
            tooltip=[
                _alt.Tooltip("variant:N"),
                _alt.Tooltip("sink:Q", format=".1%"),
                _alt.Tooltip("tokens:Q"),
            ],
        )
    )
    mo.ui.altair_chart(styled(exp_chart, h=260))
    return


# ------------------------------------------------------------ 7 · over-mixing meter (stretch)
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
                f"""### 07 · Feel the collapse  _(stretch)_

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
            mo.ui.altair_chart(styled(meter, h=260)),
        ],
        gap=0.8,
    )
    return


# ------------------------------------------------------------ 8 · model-size comparison (stretch)
@app.cell
def _(mo):
    run_compare = mo.ui.run_button(label="Compare model sizes ▸")
    mo.vstack(
        [
            mo.md(
                "### 08 · Scale it up  _(stretch)_\n\nThe paper's most demo-friendly "
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
    mo.ui.altair_chart(styled(comp_chart, h=280))
    return


# ------------------------------------------------------------ 9 · so what
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
