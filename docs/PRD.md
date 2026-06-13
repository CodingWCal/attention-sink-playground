# PRD: Attention Sink Playground

> **Product Requirements Document, v1.0**
> Author: Calvin Van
> Event: alphaXiv x marimo "Bring Research to Life: molab Notebook Competition #2"
> Paper: "Why do LLMs attend to the first token?" (Barbero et al., 2025, arXiv:2504.02732, COLM '25)
> Status: Ready for build

---

## How to use this PRD (Claude Design then Claude Code)

This document is written to flow through two tools. Read this section first so each tool knows its job.

**Stage 1, Claude Design.** The real deliverable is a marimo notebook (a Python file), not a web app. Claude Design cannot output marimo. Use Claude Design to produce a **high fidelity visual mockup and storyboard** of the notebook: the layout zones, the color and type system, the way the heatmap and charts look, the placement of controls, and the narrative reading flow from top to bottom. Treat the Design output as a "what the finished notebook should look and feel like" reference, plus reusable design tokens (colors, spacing, fonts). The sections most relevant to Claude Design are: 5, 8, 10, 13, 15, 16, and the design direction notes inside section 11.

**Stage 2, Claude Code.** Open the Design zip in Claude Code. Claude Code's job is to translate the visual mockup into a working marimo notebook in Python, mapping each designed element to a `mo.ui` component and wiring it to the PyTorch and `transformers` logic. The sections most relevant to Claude Code are: 8, 9, 11, 12, 13, 14, 15, 16, 18, and 25. The design tokens from Stage 1 should carry into the notebook's CSS and chart theming so the built notebook matches the mockup.

**One rule for both tools:** keep scope tight. This is an interpretability and visualization project on a small pretrained model. It is not a model training or paper reproduction project. When in doubt, cut features, not polish.

---

## 1. Project title

**Attention Sink Playground: Visualizing Why LLMs Attend to the First Token**

Short name for repo and links: `attention-sink-playground`.

---

## 2. One-sentence summary

An interactive marimo notebook where anyone can type a prompt and watch, live, how a Transformer dumps a large share of its attention onto the first token, turning the paper's "attention sinks help prevent over-mixing" argument into something you can see, poke, and break.

---

## 3. Problem statement

Attention sinks are one of the most reliably observed and least intuitively understood behaviors in modern LLMs. Most people who have heard of them know the surface fact ("models attend to the first token a lot") but cannot answer the interesting question: why would a model learn to waste so much attention on a token that often carries no meaning?

The paper gives a clean answer (sinks act as a release valve that stops token representations from blurring together across layers), but the answer lives in math about information propagation, Jacobian norms, and rank collapse. That is a high barrier for engineers, students, and the research-curious.

The gap this project fills: there is no hands-on, no-setup way to *feel* the attention sink phenomenon and the over-mixing intuition behind it. A reactive notebook closes that gap. You type a sentence, you see the sink, you change the conditions the paper talks about (length, depth, repetition, the first token itself), and you watch the behavior respond exactly as the theory predicts.

---

## 4. Research paper summary (beginner friendly)

**Paper:** "Why do LLMs attend to the first token?" by Federico Barbero, Alvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Petar Veličković, Razvan Pascanu, and Michael M. Bronstein. Published at COLM 2025. arXiv:2504.02732.

**The observation everyone already knows.** Large language models consistently send a big chunk of their attention to the very first token in a sequence. This is called an "attention sink." Earlier work mostly tried to either exploit this (for example, to make streaming attention work) or to get rid of it (because it complicates quantization and raises some security concerns).

**The question this paper actually answers.** Prior work described *when* sinks appear. This paper asks *why* a model would learn them in the first place, and *what they are useful for*.

**The core claim.** Sinks are a defense against "over-mixing." In a Transformer, attention is the only way tokens share information with each other. Stack many layers and feed in a long context, and tokens keep mixing with each other over and over. Past a point, every token's representation starts to look like every other token's. The signal turns to mush. Researchers call versions of this rank collapse, representational collapse, or over-squashing.

The first token, the paper argues, acts as a near-empty parking spot. When an attention head has nothing useful to do, it can dump its attention on the first token instead of forcing more mixing between meaningful tokens. That dumped attention is close to a "no-op." It does not move much information around, which keeps the other token representations distinct. So the sink is not a bug or wasted compute. It is a learned mechanism that protects the diversity of representations as depth and context grow.

**What the experiments show (the parts worth visualizing).**
- **Perturbation spread.** In Gemma 7B, changing one token's input spreads through the network much more slowly when the first-token sink is present. The sink dampens how far a small change ripples. This matches the theory: less mixing means less sensitivity.
- **Bigger and deeper models sink harder.** In the Llama 3.1 family, sink behavior gets stronger as the model gets larger. In the 405B model, a large majority of attention heads show strong sink behavior. Deeper models and longer contexts need stronger sinks to avoid collapse. This is the single most demo-friendly result, and the notebook can approximate it by comparing small versus larger models.
- **It is about position, not the special token.** Sinks tend to form at the first *position* because of where it sits, not purely because of a special beginning-of-sequence marker. That said, training choices matter: if a model is always trained with a fixed beginning-of-sequence token and that token is later removed, performance drops sharply. So the behavior is partly data and training dependent.
- **Conditional attention.** Some heads treat the first token as a "default" target and only do real work when a specific trigger pattern shows up (the paper notes things like an apostrophe pulling a head's attention away from the sink). So a head's choice to use the sink can be context dependent.

**One correction to keep front and center in the UI.** The "first token" is not the first letter. A token can be a whole word, a piece of a word (a subword), a punctuation mark, or a special start-of-sequence marker, depending on the tokenizer. The notebook must show the actual tokenization so users internalize this.

---

## 5. Target audience

Three concentric rings, in priority order.

1. **Primary: AI-curious engineers and CS students.** They know what a neural network is and can read Python, but have not internalized attention internals. They want intuition fast. The notebook should reward them within 30 seconds.
2. **Secondary: ML practitioners and applied researchers.** They know attention, maybe even know the term "attention sink," but have not played with it directly. For them, the value is the live experiments and the over-mixing meter, plus the model-size comparison.
3. **Tertiary: hackathon judges and a recruiter.** They are evaluating clarity, craft, interactivity, and whether the author understands the research. Everything should read as deliberate and polished. The recruiter screening is connected to marimo and CoreWeave, so the GPU story and the DevRel-quality explanations carry extra weight (see sections 19 and 23).

Design implication for Claude Design: the visual tone should feel like a clean, modern "research playground," approachable rather than academic. Friendly but not toy-like. Think a well-designed explorable explanation, not a dense paper figure.

---

## 6. Goals

- A user can type any prompt and immediately see the model's tokenization and an attention heatmap, with no setup and no code editing.
- A user walks away able to explain, in their own words: what a token is, what attention is, what an attention sink is, why the first token attracts so much attention, and why that matters.
- The notebook makes the paper's central claim tangible: sinks correlate with less over-mixing, and the effect strengthens with depth.
- At least one custom experiment ("what happens if we break the first token?") lets users generate their own evidence, not just read conclusions.
- The notebook uses a GPU in a way that is genuine and motivated, not bolted on, by enabling a model-size comparison and batched experiments (see section 19).
- The whole thing is reproducible: it runs in molab from a shareable link, and the source lives in a GitHub repo with pinned dependencies.
- It is polished enough to anchor a recruiter conversation and a 2 to 4 minute video.

---

## 7. Non-goals

- **Not reproducing the paper.** No re-deriving the theory, no reproducing the 405B results, no Jacobian norm experiments at scale.
- **Not training or fine-tuning a model** as part of the MVP. (A tiny toy-training demo is an optional stretch only, section 9, and is explicitly fenced off.)
- **Not a general attention visualizer** with every bell and whistle (no full BertViz clone). Stay focused on the first-token sink story.
- **Not a production web app.** It is a notebook. Resist the urge to build auth, multi-page routing, or a backend.
- **Not multi-model-zoo sprawl.** Support a small, curated set of models that are known to expose attention weights and are ungated. Quality over coverage.
- **Not mechanistic-interpretability depth** (no circuits, no probing classifiers). That is a different project.

---

## 8. MVP feature list

These are the must-haves. The notebook is not done until all of these work reliably.

1. **Prompt input.** A text area for a custom prompt, plus a dropdown of curated preset prompts that each illustrate a specific point (a normal sentence, a repeated phrase, a very short prompt, a long prompt, a prompt starting with punctuation).
2. **Run control.** A "Run" button that gates model inference so the model does not re-run on every keystroke. Cheap visualization updates (changing layer or head) happen instantly without re-running inference.
3. **Tokenized prompt display.** Render the prompt as discrete, labeled token chips so users see exactly how text splits into tokens, with the first token visibly marked.
4. **Attention heatmap.** A token-by-token heatmap for a selected layer and head, with hover tooltips showing the exact attention weight from one token to another.
5. **Layer selector and head selector.** Sliders or dropdowns. Changing either updates the heatmap and metrics instantly (no re-inference).
6. **Attention-received bar chart.** A bar chart showing how much attention each token receives (averaged over the query positions that can see it), making the first-token spike obvious.
7. **First-token attention metric.** A headline number: the percentage of attention mass going to the first token, for the current layer and head (and an "averaged over all heads" view).
8. **Layer-wise sink chart.** A line chart of average first-token attention across all layers, so users see the sink grow with depth.
9. **Explanatory text throughout.** Short, plain-language markdown cells that build the concepts in order (token, attention, sink, why, so what). Progressive disclosure: a reader can follow top to bottom without prior knowledge.
10. **The "break the first token" experiment (custom extension).** A side-by-side comparison across prompt variants (normal, strange first token, repeated words, short vs long, first token replaced with punctuation, repeated prefix) showing how first-token attention changes. This is the creativity centerpiece and is part of the MVP, not a stretch.

---

## 9. Stretch feature list

Only after the MVP is solid and polished. Each is independently optional.

1. **Over-mixing meter.** Using hidden states, compute and chart the average pairwise cosine similarity between token representations at each layer (higher similarity = more collapse). Show that representations get more similar in deeper layers, and let users see how this interacts with the sink. This is the most paper-faithful stretch and the highest value one. (It is listed as stretch only because it adds a second tensor extraction path; if time allows, promote it.)
2. **Model-size comparison (GPU).** Load two or three models of increasing depth from the same family (for example DistilGPT-2, GPT-2, GPT-2-large, GPT-2-xl) and overlay their layer-wise sink curves to visualize the "bigger and deeper sinks harder" finding. This is the primary justification for GPU use (section 19).
3. **Model selector.** A dropdown to switch among a curated, ungated set of models (GPT-2 family, Pythia family, optionally Qwen2.5-0.5B or TinyLlama). Keep the list short and known-good.
4. **Prepend or replace the first token.** A control to prepend a beginning-of-sequence token (or a chosen token) and watch the sink move or change strength, directly probing the "position vs special token" question.
5. **Max token length control.** A slider to truncate or cap prompt length, to demonstrate the context-length effect on sinks.
6. **Perturbation spread mini-demo.** Change one input token and visualize how much downstream token representations shift with vs without a strong first-token sink. This approximates the paper's Gemma perturbation experiment in spirit.
7. **Toy training demo (fenced, highest risk, lowest priority).** Train a tiny Transformer on a toy language-modeling task and watch whether sink behavior emerges over training steps. Only attempt if everything else is finished and stable. Keep it in a clearly separated, optional section so it can be cut without affecting the rest. Do not let this become the project.

**Scope guardrail:** ship the MVP plus stretch items 1 and 2 for the strongest submission. Treat everything else as bonus.

---

## 10. User journey through the notebook

A first-time reader scrolls top to bottom. The journey is designed as a guided explorable, not a reference manual.

1. **Land.** A title, a one-line hook, and a single sentence on what they are about to do. A preset prompt is already loaded and a heatmap is already showing, so the page is alive on arrival (no blank state).
2. **Meet a token.** They read two sentences on what a token is, then see the loaded prompt split into token chips. The first token is highlighted. The "first token is not the first letter" correction lands here.
3. **Meet attention.** Two sentences on what attention is (each token decides how much to listen to each earlier token), then the heatmap. They hover and see weights.
4. **See the sink.** The attention-received bar chart and the headline first-token percentage make the spike undeniable. A sentence names it: this is an attention sink.
5. **Ask why.** A short, plain explanation of the over-mixing idea (the telephone / classroom-notes analogy), framed as the paper's answer.
6. **Watch it grow with depth.** The layer-wise sink chart. They drag the layer slider and watch the heatmap and number change, seeing the sink strengthen in deeper layers.
7. **Break it.** They reach the experiment section, run the variant comparison, and generate their own evidence for how length, repetition, and the first token itself change the sink.
8. **(Stretch) Feel the collapse.** The over-mixing meter ties the loop shut: sinks down, similarity up.
9. **(Stretch) Scale it up.** The model-size comparison shows the effect getting stronger in bigger models, on a GPU.
10. **Leave with the point.** A closing "so what" cell: sinks are a learned defense mechanism, this is why interpretability of "boring" patterns matters, and a link back to the paper.

Design implication for Claude Design: this ordering *is* the page layout. Storyboard it as a vertical narrative with clear section breaks, where interactive controls sit right next to the visual they affect.

---

## 11. Technical implementation plan

**Framework.** marimo reactive notebook, run and shared via molab. Single source file `notebook.py`.

**The reactive architecture (this is the most important engineering decision).** Separate expensive work from cheap work so the UI stays snappy:

- **Cell A, load model (cached).** Load tokenizer and model once and cache them. Switching layer or head must never reload the model. Use a module-level singleton or marimo caching so re-runs are free.
- **Cell B, run inference (gated).** Runs only when the "Run" button is clicked or the prompt is submitted via a form. Produces the attention tensors and hidden states for the current prompt and stores them. This is the only expensive step per prompt.
- **Cell C, compute metrics (cheap, reactive).** Derives first-token percentages, per-token received attention, and the layer-wise curve from the already-computed tensors. Re-runs freely.
- **Cell D, render visuals (cheap, reactive).** Renders the heatmap and bar chart for the currently selected layer and head. Re-runs instantly when sliders move because it only indexes into existing tensors.

This separation is what makes the notebook feel like a playground instead of a laggy form. It also demonstrates that the author understands reactive notebook design, which matters for the DevRel and recruiter angle.

**Model loading, the one critical gotcha.** Modern `transformers` defaults to attention implementations (SDPA / Flash) that return `None` for attention weights. You must load with eager attention or you will get no heatmap:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",   # REQUIRED so attentions are returned, not None
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)
model.eval()
```

**Inference and extraction.**

```python
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs, output_attentions=True, output_hidden_states=True)

attentions = out.attentions            # tuple: num_layers x (batch, heads, seq, seq)
hidden_states = out.hidden_states      # tuple: (num_layers+1) x (batch, seq, dim)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
```

**First-token attention metric (mind the causal mask).** Token 0 can only attend to itself, so its row trivially puts 100% on the first token. Exclude position 0 from summaries or you will overstate the sink:

```python
A = attentions[layer][0, head]        # (seq, seq); row = query, col = key
to_first_per_query = A[:, 0]          # attention each query puts on token 0
avg_to_first = A[1:, 0].mean().item() if A.shape[0] > 1 else 1.0  # skip trivial row 0
```

**Per-token received attention** (for the bar chart): average each column over the query rows that are allowed to see it (rows at or after that key index), to avoid diluting with masked zeros.

**Layer-wise sink curve:**

```python
layerwise = [attentions[L][0, :, 1:, 0].mean().item() for L in range(len(attentions))]
```

**Over-mixing meter (stretch):**

```python
import torch.nn.functional as F
H = hidden_states[layer][0]           # (seq, dim)
Hn = F.normalize(H, dim=-1)
sim = Hn @ Hn.T                       # (seq, seq) cosine similarities
# mean off-diagonal similarity = collapse score (higher = more over-mixed)
```

**Model-size comparison (stretch, GPU).** Loop over a small list of same-family models, compute the layer-wise curve for each on the GPU, and overlay them. Normalize the x axis to relative depth (0 to 1) so models of different depths line up.

**Design direction notes for Claude Design (carry into the build):**
- One restrained accent color used for "the sink" so it reads as the protagonist throughout (suggest a warm signal color like amber or a clear blue; pick one and use it consistently).
- Heatmap colormap: a clean sequential scale (suggest viridis or blues) with the diagonal and first column legible. Tooltips on hover.
- Token chips: monospace, subtle border, the first token visually distinct (filled accent). Optionally tint each chip by how much attention it receives.
- Generous vertical rhythm and clear section dividers, matching the narrative in section 10.
- Type: a clean sans for body, monospace for tokens and code. Keep it minimal and editorial.

---

## 12. Suggested Python libraries

| Library | Purpose | Notes |
|---|---|---|
| `marimo` | Notebook framework and UI | Provides `mo.ui.*`, layout, markdown |
| `torch` | Model inference | Pre-installed on molab; use GPU when available |
| `transformers` | Pretrained models and tokenizers | Load with `attn_implementation="eager"` |
| `altair` | Interactive heatmap and bar charts | Idiomatic in marimo, interactive, shareable. Primary choice |
| `numpy` | Array math for metrics | Lightweight |
| `pandas` or `polars` | Shaping data for Altair | Altair expects tidy long-form data |

Optional or fallback:
- `matplotlib` for a quick static heatmap if Altair theming gets fiddly.
- `plotly` as an alternative interactive heatmap (good hover), if preferred over Altair.

Keep the dependency list short. Pin versions for reproducibility (see section 22).

---

## 13. marimo UI components needed

Use these `mo.ui` elements. All are reactive: dependent cells re-run when the value changes (subject to the gating in section 11).

- `mo.ui.text_area(...)` for the custom prompt.
- `mo.ui.dropdown(options=..., value=...)` for preset prompts (use a dict mapping a friendly label to the prompt string).
- `mo.ui.run_button(label="Run")` to gate inference. Read `.value` to know it was clicked.
- `mo.ui.slider(start, stop, step=1)` or `mo.ui.dropdown` for layer selection.
- `mo.ui.slider(start, stop, step=1)` or `mo.ui.dropdown` for head selection.
- `mo.ui.switch(label="Long prompt")` and `mo.ui.switch(label="Repeated prompt")` for the quick toggles.
- `mo.ui.dropdown` for the optional model selector (stretch).
- `mo.ui.number(...)` for the optional max-token-length control (stretch).
- `mo.ui.text(...)` for the optional "prepend or replace first token" control (stretch).

Layout and presentation:
- `mo.md(f"...")` for all explanatory text, with UI elements embedded via f-strings.
- `mo.hstack([...])` and `mo.vstack([...])` to place controls beside the visual they affect.
- `mo.ui.tabs({...})` to separate the main playground from the experiment section and the stretch sections, keeping the page tidy.
- `mo.callout(kind="info")` for the key takeaways and the "first token is not the first letter" correction.
- `mo.stat(...)` (or a styled `mo.md`) for the headline first-token percentage.
- Consider `mo.ui.batch(...).form()` as an alternative to the run button if you want all prompt controls submitted together.

Design implication for Claude Design: mock each of these as a concrete control with a label and state, positioned next to its output. The mockup should make the control-to-visual relationship obvious.

---

## 14. Data and model requirements

**No external dataset is required for the MVP.** Inputs are user-typed prompts plus a small curated set of preset prompts shipped with the notebook.

**Curated preset prompts** (ship these; each illustrates a specific point):
- A normal sentence (baseline sink).
- A heavily repeated phrase (to show repetition effects).
- A very short prompt (two or three tokens).
- A long prompt (to show context-length effects).
- A prompt that starts with punctuation or an unusual token.
- A prompt with a repeated prefix token.

**Models.** Use small, ungated, attention-exposing causal LMs:
- **Default (CPU-safe):** `distilgpt2` (~82M) or `gpt2` (~124M). Fast, reliable, exposes attentions.
- **Larger (GPU):** `gpt2-large` (~774M), `gpt2-xl` (~1.5B), `EleutherAI/pythia-410m`, `EleutherAI/pythia-1.4b`.
- **Optional alternatives:** `Qwen/Qwen2.5-0.5B`, `TinyLlama/TinyLlama-1.1B`.

**Avoid in the default path:** gated models such as Llama 3.x and Gemma. They require accepting a license and an auth token, which adds friction in molab and can break a live demo. If you reference them (the paper uses Llama and Gemma), do it as discussion, not as a default code path. Flag the gating in the UI if you do add them.

**Tokenizer note to surface in the UI.** GPT-2 does not prepend a beginning-of-sequence token by default, so the "first token" is simply the first real token of the prompt. Models like Llama and Gemma prepend a special start token. This difference is exactly the "position vs special token" point from the paper and is worth a sentence in the notebook.

**Compute.** molab gives 4 CPUs and 32 GB RAM by default, with an optional NVIDIA RTX Pro 6000 Blackwell GPU (96 GB VRAM) attachable from the notebook specs button. The MVP runs fine on CPU. The GPU is for the size-comparison and batched experiments (section 19). Sessions run up to 12 hours, which is more than enough for a demo.

---

## 15. Visualizations needed

1. **Tokenized prompt strip.** Token chips, first token highlighted, optional attention tint. (Rendered HTML, not a chart.)
2. **Attention heatmap.** Token-by-token, selected layer and head, hover tooltips with exact weights. The visual centerpiece.
3. **Attention-received bar chart.** One bar per token, height = attention received, first token bar visually dominant.
4. **First-token attention metric.** A large headline number (percentage), for the current head and an all-heads average.
5. **Layer-wise sink line chart.** Average first-token attention per layer, showing growth with depth.
6. **(Stretch) Over-mixing meter.** Average pairwise cosine similarity per layer, ideally overlaid against the sink curve to show the relationship.
7. **(Stretch) Model-size comparison.** Overlaid layer-wise sink curves for two or three models against relative depth.
8. **(Stretch) Experiment comparison chart.** A grouped bar chart of first-token attention across the prompt variants in the "break it" section.

Visual consistency: every chart shares the design tokens from Stage 1 (colors, fonts, the single accent for "the sink").

---

## 16. Notebook section-by-section outline

This is the build order and the on-page structure. Each numbered item is roughly one logical block of cells.

1. **Title and hook.** Title, one-line hook, one sentence on what the reader will do.
2. **Setup (collapsed or quiet).** Imports, device detection, cached model load. Keep visually minimal; this is plumbing.
3. **"What is a token?"** Two sentences, then the tokenized prompt strip with the first token highlighted, then the correction callout.
4. **"What is attention?"** Two sentences, then the controls (prompt area, preset dropdown, run button) and the heatmap side by side.
5. **"There's a sink."** The attention-received bar chart and the headline first-token percentage, with one sentence naming the phenomenon.
6. **"Why would a model do this?"** The over-mixing explanation in plain language (telephone / classroom analogy), framed as the paper's answer.
7. **"It grows with depth."** Layer and head selectors plus the layer-wise sink chart. Encourage dragging the layer slider.
8. **Experiment: "What happens if we break the first token?"** The variant comparison and grouped bar chart. The custom extension.
9. **(Stretch) "Feel the collapse."** The over-mixing meter, overlaid with the sink curve.
10. **(Stretch) "Scale it up."** The model-size comparison on GPU.
11. **"So what?"** Closing takeaways: sinks are a learned defense, why this matters, link to the paper and to your repo.
12. **Appendix.** Model list, references, reproduction notes, credits.

---

## 17. Evaluation criteria (is the project successful?)

The project is successful if all of the following are true.

- **Runs from a cold molab link** with no manual setup, and a stranger can use it without reading code.
- **Alive on arrival:** a heatmap and metrics show on load, no blank state.
- **Snappy interaction:** moving the layer or head selector updates visuals near-instantly (because inference is gated and tensors are cached). No multi-second lag on a slider drag.
- **The sink is undeniable:** within 30 seconds, a new user can see and name the first-token spike.
- **Comprehension:** a non-expert can correctly explain token, attention, sink, why, and so-what after going through it once.
- **The experiment produces real evidence:** the variant comparison visibly changes first-token attention in ways consistent with the paper.
- **Reproducible:** the source lives in GitHub with pinned dependencies, and the notebook runs the same on a fresh session.
- **Polished:** consistent design, no broken cells, no stray debug output, clear writing, no typos.
- **GPU use is genuine** (if the stretch is included): the size comparison or batched experiment actually uses the GPU and is motivated by the science, not decoration.

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Attention weights come back as `None` | High if unaware | Load with `attn_implementation="eager"`; assert attentions are not None right after load |
| Notebook feels laggy (re-inference on every keystroke) | High if naive | Gate inference behind a run button or form; cache the model; keep slider-driven cells cheap (index into existing tensors) |
| First-token percentage looks like 100% everywhere | Medium | Exclude the trivial query position 0 from summaries (causal mask makes its row self-attend fully) |
| Gated model breaks the live demo | Medium | Default to ungated GPT-2 / Pythia; treat Llama/Gemma as discussion only |
| Large model OOMs or is slow on CPU | Medium | Keep MVP on small models; only use large models when GPU is attached; use bfloat16 on GPU |
| Heatmap unreadable for long prompts | Medium | Cap displayed sequence length, or downsample / scroll; warn on very long prompts |
| Altair theming eats time | Low to Medium | Fall back to matplotlib for a clean static heatmap if needed; do not let charting polish block the MVP |
| Scope creep (toy training, model zoo) | High | Hard guardrail: MVP plus stretch 1 and 2 only; everything else is bonus |
| Tokenizer confusion ("first token = first letter") | Medium | Always show real tokenization; include the correction callout prominently |
| Over-claiming the science | Low | Frame the over-mixing meter as an illustration consistent with the paper, not a reproduction of its proofs |

---

## 19. Hackathon judging alignment

The competition is the alphaXiv x marimo molab Notebook Competition #2. The jury favors notebooks that give an intuitive understanding of a research result through code, UI, and text, and that add a custom extension that improves on or gives insight into the paper. The competition explicitly runs on GPUs this round, and Transformer inference is named as in-scope. Mapping the rubric to this project:

- **Paper engagement.** The notebook is built directly on the paper's central claim (sinks prevent over-mixing) and visualizes three of its findings: the sink itself, the depth effect, and (via the over-mixing meter) the collapse the sink defends against. The closing section ties everything back to the paper.
- **Creativity and impact.** The "break the first token" experiment lets users generate their own evidence, and the over-mixing meter makes an abstract failure mode visible. The framing turns a "boring" pattern into a memorable lesson.
- **Interactivity and workflow.** Reactive controls update visuals live, with a deliberately gated inference step so it stays fast. This is exactly what marimo is for, and it shows the author used the framework idiomatically.
- **Design and shareability.** A clean, narrative, explorable layout (carried over from the Claude Design mockup), shareable as a no-login molab link. Strong screenshots and a tight video.
- **Code quality.** Clear cell separation, cached model load, no dead cells, pinned deps, a real README. The one critical gotcha (eager attention) handled correctly.
- **GPU utilization.** This is where the size-comparison stretch earns its place. molab's GPUs run on CoreWeave hardware, and using one to compare DistilGPT-2 through GPT-2-xl directly visualizes the paper's "bigger and deeper models sink harder" finding. The GPU is motivated by the science, not bolted on. Batched variant experiments can also run on the GPU. (This also connects to the recruiter angle in section 23.)

**Submission logistics.** Submit a molab link plus a video explainer through the competition form. Up to 3 submissions per person, solo or group.

**Deadline, flag this.** The official competition page states the cutoff is **June 28, 11:59 PM PST**. Your brief said June 29. The gap is likely a timezone artifact: 11:59 PM PST on June 28 is roughly 2:59 AM EST on June 29 in Jacksonville. Plan to submit before end of day June 27 to leave a buffer, and do not bank on the 29th. The winners event is July 9.

---

## 20. Video demo script outline (2 to 4 minutes)

Keep it tight, show the live notebook, let the interactivity carry it.

1. **Hook (0:00 to 0:20).** "Every large language model has a strange habit: it pours a huge amount of attention onto its very first token, even when that token means nothing. Why?" Show the title and the live heatmap already on screen.
2. **Token and attention in 20 seconds (0:20 to 0:50).** Point at the token chips ("text becomes tokens, and the first token is not the first letter"), then the heatmap ("each token decides how much to listen to earlier ones").
3. **The sink (0:50 to 1:20).** Show the bar chart and the headline percentage spiking on the first token. Name it: this is an attention sink.
4. **The why (1:20 to 1:50).** One clean line of the over-mixing intuition (telephone / everyone copying notes). "The paper's answer: the sink is a release valve that stops representations from blurring together."
5. **Make it move (1:50 to 2:30).** Drag the layer slider and narrate the sink getting stronger with depth. Run the "break the first token" experiment and show first-token attention shift across variants. This is the wow moment.
6. **(If included) Scale and collapse (2:30 to 3:10).** Show the over-mixing meter, then the GPU model-size comparison: bigger models sink harder, matching the paper. Mention it ran on a molab GPU (CoreWeave).
7. **Takeaway and link (3:10 to 3:40).** "Attention sinks are not a bug. They are a learned defense mechanism, and you can see it for yourself." Show the molab link and the repo. Done.

Tips: record at a comfortable pace, keep the screen uncluttered, no dead air while a cell runs (pre-warm the model before recording), and let the live interaction be the star.

---

## 21. Development timeline (MVP to polished submission)

Anchored to a June 27 internal target (one-day buffer before the June 28 PST cutoff). Adjust to your actual availability; this assumes part-time evenings and a weekend.

- **Day 1, foundation.** molab notebook created, model loads with eager attention, inference returns attentions, tokens render. Smallest possible heatmap working. (De-risk the one critical gotcha first.)
- **Day 2, core playground.** Prompt input, run button gating, layer and head selectors, heatmap and bar chart reactive and fast. First-token metric and layer-wise curve working.
- **Day 3, narrative and experiment.** Write all explanatory cells in plain language. Build the "break the first token" experiment and its comparison chart. The MVP is now feature-complete.
- **Day 4, stretch (pick highest value).** Add the over-mixing meter, then the GPU model-size comparison if time allows. Attach a GPU in molab and verify it actually runs there.
- **Day 5, polish.** Apply the Claude Design tokens (colors, fonts, spacing), tidy every cell, remove debug output, proofread all text, cap long-prompt edge cases, test a cold-start run from scratch.
- **Day 6, ship.** Record the video, take screenshots, write the README, push to GitHub, confirm the molab link works for a logged-out viewer, submit the form. Keep a day in reserve.

If time gets tight, cut in this order: toy training, perturbation demo, model selector, max-token control, then the size comparison. Never cut: the MVP and the writing.

---

## 22. README and GitHub structure

marimo notebooks are single Python files, so the repo stays simple. Keep GitHub as the source of truth and let molab mirror it.

```
attention-sink-playground/
├── notebook.py            # the marimo notebook (single source of truth)
├── pyproject.toml         # pinned dependencies (or use inline PEP 723 metadata in notebook.py)
├── README.md              # what it is, how to run, molab link, screenshots
├── assets/
│   ├── heatmap.png        # screenshot for the README and submission
│   └── thumbnail.png      # video thumbnail
├── data/
│   └── preset_prompts.py  # curated prompts (or .json)
├── LICENSE                # MIT is fine
└── .gitignore
```

**Dependency pinning.** Either a `pyproject.toml` with pinned versions, or PEP 723 inline script metadata at the top of `notebook.py` (marimo and molab both understand this and it keeps the notebook self-contained for reproducibility).

**README contents (in order):**
1. One-line description and a hero screenshot of the heatmap.
2. A prominent "Open in molab" link (and the note that no login is needed to view).
3. What the paper is and the one-sentence finding, with a link to arXiv:2504.02732.
4. What the notebook does and what a reader will learn.
5. How to run: in molab (link), or locally via `marimo edit notebook.py`, or open the molab URL with `marimo edit <url>`.
6. The model list and the eager-attention note.
7. Credits (paper authors, the competition) and license.

Keep the README skimmable. Lead with the visual and the link.

---

## 23. Recruiter-screening talking points

The screening connects to marimo and CoreWeave, so lean into research understanding, tool craft, communication, and the GPU/infra story.

- **Research literacy.** "I read the paper and built around its actual thesis, not just the surface fact. The interesting claim is that sinks are a defense against over-mixing, where token representations collapse into each other across depth. I made that claim visible." Be ready to explain over-mixing, why depth and context length make it worse, and the position-vs-special-token nuance.
- **Translating research to a tool.** "I turned a math-heavy result into a 30-second intuition for a non-expert. The hardest design problem was ordering the concepts so a beginner could follow token to attention to sink to why, without front-loading jargon."
- **Reactive engineering judgment.** "The naive version re-runs the model on every keystroke and feels broken. I separated expensive inference (gated behind a run button, model cached) from cheap visualization (indexing into existing tensors), so dragging a slider is instant. That separation is the whole reason it feels like a playground." This shows you understand marimo's reactivity model, which matters to the marimo team specifically.
- **The CoreWeave / GPU angle.** "molab runs on CoreWeave GPUs, so I used one to compare models from DistilGPT-2 up to GPT-2-xl and show the sink strengthening with depth, which reproduces the paper's qualitative finding that bigger and deeper models sink harder. The GPU is doing real, motivated work, not sitting idle for show." You can speak to attaching the RTX Pro 6000 Blackwell, batching the experiments, and bfloat16 inference.
- **DevRel instinct.** "I wrote it the way I'd write a tutorial: a hook, progressive disclosure, an interactive payoff, and a clear takeaway, plus a tight video and a clean repo." Point to the writing quality and the explorable structure as evidence.
- **Honest scoping.** "I deliberately kept it as interpretability on a pretrained model rather than training or reproduction, because the goal was a reliable, polished, understandable artifact under a deadline. I know where the line was and I held it." This signals product judgment, which is often what a screening is really probing.

Have the live notebook and the video ready to screen-share. Let them poke it.

---

## 24. Explain it like I'm 12

Imagine a classroom where every student is taking notes. Every few minutes, the teacher says "okay, now copy a bit from your neighbors and mix it into your own notes." A little of this is helpful: you fill in things you missed. But if everyone keeps copying everyone else over and over, something bad happens. After enough rounds, every student's notes look exactly the same: a blurry soup. Nobody has their own clear idea anymore. That blurring is what researchers call "over-mixing."

A language model is a bit like that classroom. The words (well, tokens) keep mixing information with each other, layer after layer. Too much mixing and all the words start to look the same to the model, which is bad.

So the model learns a clever trick. It picks a spot to dump attention it does not need, like a student who, when there's nothing useful to copy, just doodles in the margin instead of muddying their real notes. The model uses the very first token as that doodle spot, the "parking space," the sink. By sending spare attention there, the model avoids over-mixing the words that actually matter, and everyone gets to keep their own clear notes.

One thing that trips people up: the "first token" is not the first letter. The model chops your sentence into chunks called tokens (a chunk might be a whole word, part of a word, or a punctuation mark), and it's the first chunk that becomes the parking space.

That's the whole idea this notebook lets you see: type a sentence, and watch the model park a big pile of attention on that first chunk.

---

## 25. Open questions and assumptions to verify

Flagging these honestly rather than pretending. Resolve the ones marked build-blocking before or during the build; the rest are nice-to-confirm.

**Build-blocking or near-blocking:**
- **Eager attention on every chosen model.** Confirm each model in your final list actually returns attentions with `attn_implementation="eager"` in the installed `transformers` version. (GPT-2 and Pythia are reliable; verify any others.)
- **Causal-mask handling in the metric.** Confirm the decision to exclude query position 0 from first-token summaries, and apply it consistently across the headline number, the bar chart, and the layer-wise curve, so they tell a coherent story.
- **molab GPU availability and model fit.** Confirm a GPU can be attached in your molab session and that your largest intended model loads in 96 GB VRAM with headroom. (It will; just verify before relying on it in the video.)
- **Reactivity gating actually works.** Confirm that moving the layer or head slider does not trigger re-inference, and that the model is not reloaded on every interaction.

**Worth confirming for accuracy (mostly affects how you describe things, not whether it runs):**
- **Per-token "received attention" definition.** Decide and document whether "received" means a masked mean over valid query rows or a raw column sum, and make the bar chart's y-axis label honest about it.
- **How faithfully the over-mixing meter maps to the paper.** The cosine-similarity meter is an illustration of representational collapse, not a reproduction of the paper's formal measures. Frame it that way in the text so you do not over-claim.
- **The "position vs special token" demonstration.** If you add the prepend/replace-first-token control, confirm the behavior you show (sink moving with position, or strengthening with a special start token) is described accurately for the specific model used, since GPT-2 and Llama/Gemma differ here.
- **Model-size comparison fairness.** When overlaying curves across model sizes, confirm you normalize by relative depth and note that cross-family comparisons (for example GPT-2 vs Pythia) are not perfectly apples-to-apples.

**Assumptions baked into this PRD:**
- The submission is a molab link plus a video; no formal paper reproduction is expected.
- Small, ungated pretrained models are sufficient to demonstrate the phenomenon (they are; the sink shows clearly even in GPT-2).
- The GPU is a bonus and the MVP must work on CPU, so a GPU outage on demo day cannot sink the submission.
- The June 28 11:59 PM PST cutoff is authoritative over the June 29 figure in the brief (verify on the competition page before the deadline in case dates shift).

---

*End of PRD. Hand this to Claude Design for the visual mockup, then open the design in Claude Code to build `notebook.py`.*
