# molab Deployment & Submission Checklist

Target: submit a public molab link (plus a short video) to the **alphaXiv × marimo
"Bring Research to Life" molab Notebook Competition #2**.

> **Deadline: June 28, 11:59 PM PST.** Aim to submit by **June 27** for a buffer.
> Confirm the exact date/time on the live submission form before you rely on it —
> the public event page has at times shown the previous competition's date.

---

## 0 · Pre-flight (local, before touching molab)

- [ ] `git status` is clean; latest is committed and (optionally) pushed to GitHub.
- [ ] Headless smoke test passes:
      `python -m marimo export html notebook.py -o /tmp/check.html` → exit 0, no traceback.
- [ ] Open locally once and click through every section:
      `marimo edit notebook.py` (or `marimo run notebook.py`).
- [ ] Confirm the 5 build-blocking assumptions (PRD §25):
  - [ ] Attentions are **not None** (eager attention in effect).
  - [ ] Moving the layer/head slider does **not** re-run inference (watch it stay instant).
  - [ ] First-token % is not a flat 100% everywhere (query row 0 is excluded).
  - [ ] The model loads once and is cached.
  - [ ] The default preset renders a heatmap **on arrival** (no blank state).

## 1 · Create the molab notebook

molab = hosted marimo at <https://molab.marimo.io> (runs on marimo/CoreWeave infra).

- [ ] Sign in at <https://molab.marimo.io>.
- [ ] Create a new notebook and replace its contents with `notebook.py`
      (the PEP 723 `# /// script` header carries the dependencies — molab reads it).
      - Easiest path: locally run `marimo edit <your-molab-url>` to push the file, **or**
        copy–paste the full `notebook.py` into a fresh molab notebook.
- [ ] First run downloads GPT-2 (~500 MB) from Hugging Face. Let it finish once.
- [ ] Verify a clean cold run top-to-bottom (see §3 checklist).

## 2 · (Optional but recommended) Attach a GPU for the size comparison

Competition #2 provides GPUs — using one is a judging plus and is genuinely motivated here.

- [ ] In the molab notebook, open the runtime/specs control and **attach a GPU**.
- [ ] In the **"08 · Scale it up"** cell, extend the list:
      `COMPARE_MODELS = ["distilgpt2", "gpt2", "gpt2-large", "gpt2-xl"]`
- [ ] Click **Compare model sizes ▸** and confirm the deeper models sink harder.
      (`gpt2-xl` is ~1.5 B params — fine on the GPU, slow and RAM-heavy on CPU, so only
      add the large models once a GPU is attached.)
- [ ] No code change needed for precision — bf16 is used automatically on CUDA.

## 3 · Cold-start verification on molab (the judging rubric, as a checklist)

Run from a fresh molab session and confirm each (mirrors PRD §17):

- [ ] **Runs from a cold link** with no manual setup.
- [ ] **Alive on arrival** — heatmap + stats show on load, no blank state.
- [ ] **Snappy** — dragging layer/head updates visuals with no multi-second lag.
- [ ] **The sink is undeniable** within ~30 s (bar chart + headline %).
- [ ] **Break-the-first-token** experiment runs and the variants visibly differ.
- [ ] **Over-mixing meter** renders (blue sink vs red collapse, independent axes).
- [ ] **No stray output** — no debug prints, no error cells, no broken charts.
- [ ] Try one **edge prompt** (very long, punctuation-first, a single word) — no crash;
      long prompts show the "first N tokens" truncation note.

## 4 · Make it shareable

- [ ] Set the molab notebook to **public / link-sharing** so a logged-out viewer can open it.
- [ ] Open the share link in a **private/incognito window** to confirm no-login viewing works.
- [ ] Paste the final link into `README.md` (replace the `_(link coming soon)_` placeholder)
      and into the GitHub repo description. Commit + push.

## 5 · Screenshots & video

- [ ] Refresh `assets/heatmap.png` if desired (`python scripts/make_hero.py`).
- [ ] Grab 2–3 clean screenshots (hero, the sink bar chart, the over-mixing meter).
- [ ] Record a **2–4 min** screen demo following the shot list in **PRD §20**.
      Pre-warm the model before recording so there's no dead air while a cell runs.

## 6 · Submit

- [ ] Open the competition submission form (linked from
      <https://marimo.io/pages/events/notebook-competition>).
- [ ] Submit the **molab link** (+ video link), with **all team member names**.
- [ ] You may submit **up to 3 entries** total — keep one slot in reserve.
- [ ] Submit by **June 27** (buffer before the June 28 11:59 PM PST cutoff).

---

## Rollback / gotchas

- **Attentions come back `None`** → the model wasn't loaded with
  `attn_implementation="eager"`. The notebook sets this; don't remove it.
- **transformers version differences** → loading is version-robust (tries `dtype`
  then `torch_dtype` on GPU; uses the float32 default on CPU). If a gated model is
  ever added, it will need a HF token — stick to the ungated defaults for the demo.
- **Local `marimo run` crash on Windows/OneDrive** (torch circular import at kernel
  start) is a local-only flake mitigated by `.venv/.../sitecustomize.py`. molab
  (Linux) is unaffected; it never appears in `marimo export`.
