# Video Script: Attention Sink Playground

**Duration:** 2:30–3:30 (target the lower end for tightness).  
**Delivery:** Calm, conversational pace. Let the visuals breathe — don't rush.  
**Setup:** molab notebook open in a browser, pre-warmed (model loaded, default prompt ready).  
**Quality:** Screen + voice (no webcam needed). Capture at 1080p+; audio clear.

---

## [0:00–0:20] Hook

**SHOT:** Hero section (title, three stats).

> Every large language model has a strange habit: it pours a huge amount of attention 
> onto its very first token, even when that token means nothing. Today, you're going to 
> see it happen live. Why? And what does it tell us about how these models actually work?

*Pause. Let the question hang.*

---

## [0:20–0:45] Token and attention in 30 seconds

**SHOT:** Scroll to the token strip (section 01).

> First, what's a token? A model doesn't see letters or words. It breaks your text into 
> chunks — sometimes a whole word, sometimes a piece, sometimes just punctuation. The 
> first token — highlighted in blue — is the one everything's going to point to.

**SHOT:** Point to/highlight the first blue token.

> Notice: the first token is *not* the first letter. That space symbol marks a leading space 
> in GPT-2's tokenizer. This detail matters.

**SHOT:** Scroll to the heatmap (section 02).

> Now, attention. Each token looks at every earlier token and decides how much to listen to 
> each one. A darker blue means "I'm paying close attention to this one." Read a row left to 
> right — that's where one token sends its attention.

**SHOT:** Hover over a cell in the heatmap to show the tooltip.

> You see the numbers on hover. Watch the leftmost column — it's going to glow.

---

## [0:45–1:15] The sink (the wow moment)

**SHOT:** Scroll to section 03 (bar chart + headline %).

> Here's the weird part. Total up the attention each token receives, and one bar towers 
> over the rest.

**SHOT:** Let the eye rest on the first bar (token 0) for 2 seconds.

> That spike on the first token is called an **attention sink**. It's not random. It's not 
> a bug in the model. It's **learned behavior**.

**SHOT:** Show the headline percentage (e.g., "86% of attention from every other token 
lands on token 0, at this head").

> In this case, 86% of the attention from every other token in the sentence lands on 
> that first token. That's not a coincidence.

---

## [1:15–1:50] Why the model does this

**SHOT:** Scroll to section 04 (text explanation).

> So why would a model learn to waste so much attention on a meaningless token? Here's 
> the insight from a recent paper.

> Imagine a classroom where everyone keeps copying notes from their neighbors. A little 
> bit helps — you fill in what you missed. But copy round after round and what happens? 
> Every page looks the same. Every student's notes become a blurry soup. Nobody has their 
> own distinct ideas anymore.

**SHOT:** Let this land.

> Language models mix information the same way. Layer after layer, tokens keep mixing 
> with each other until their representations start to blur together. The model learned a 
> trick: use the first token as a parking spot. Dump the attention you don't need to use 
> there instead of forcing it to mix with the tokens that actually matter. The sink is a 
> **release valve** that keeps meaningful tokens distinct. It's a learned defense.

---

## [1:50–2:20] Watch it grow with depth

**SHOT:** Scroll to section 05 (layer-wise sink curve + slider).

> One more thing: drag the layer slider and watch what changes.

**ACTION:** Slowly drag the layer slider from left (layer 0) to right (layer 11).

> See? The sink gets stronger as you go deeper into the network. The deeper the model 
> goes, the harder it leans on that first-token parking spot. That's exactly what the 
> theory predicts: **deeper models need stronger sinks to stop token representations from 
> collapsing into each other.**

**SHOT:** Rest on a deep layer (e.g., layer 10–11).

---

## [2:20–2:50] Break it and run the experiment

**SHOT:** Scroll to section 06 (experiment button).

> But here's the best part: you can break it on purpose.

> I ran six variants of the prompt: normal, repeated words, short, long, starting with 
> punctuation, and with a repeated prefix. The question: does the sink move or change?

**ACTION:** Click "Run the experiment ▸".

*Let it run. Don't fill the silence — this is where the model does real work.*

**SHOT:** The experiment bar chart renders.

> Look. The short prompt sinks the hardest — almost 90%. The long prompt sinks the 
> least — around 57%. Repetition and weird structure move the needle too. The sink is a 
> *position effect*, not magic about a special word. You just generated your own evidence 
> for why the paper's theory makes sense.

---

## [2:50–3:20] (Optional: Over-mixing meter and scale-up)

*If you have time, include one or both of these. If running tight, skip to the closer.*

### A. Over-mixing meter (30 sec)

**SHOT:** Scroll to section 07 (dual-axis chart).

> This meter shows the other side of the coin: how similar token representations become 
> as they mix.

**SHOT:** Point to the red dashed line.

> As depth increases, tokens become more similar — they're blurring together. That's 
> representational collapse. And as that collapse pressure builds, you see the blue line — 
> the sink — climbing harder. The model is *responding* to the threat. That's the 
> relationship the paper discovered.

### B. Model-size comparison (30 sec)

**SHOT:** Scroll to section 08, click "Compare model sizes ▸".

*Let it load.*

**SHOT:** The two-model curve overlay renders.

> Finally: **bigger models sink harder.** This is the paper's most demo-friendly result. 
> DistilGPT-2 has 6 layers. GPT-2 has 12. Same prompt, but with depth normalized so 
> they line up.

**SHOT:** Point to where GPT-2 (blue) peaks higher than DistilGPT-2 (grey).

> The deeper model reaches a higher peak and sustains a stronger sink across more layers. 
> On a GPU, you can add even bigger models — GPT-2-large, GPT-2-XL — and watch the 
> effect get even more pronounced. Bigger, deeper, stronger sink.

---

## [3:10–3:30] Takeaway and link

**SHOT:** Scroll to section 09 ("So what?").

> So what? Attention sinks aren't wasted compute. They're not a quirk to patch away. 
> They're a **learned defense mechanism** against over-mixing — something the model had to 
> figure out because depth and long contexts created a real problem.

> The lesson: even the "boring," ever-present patterns inside a model can be doing 
> essential work. That's why interpretability of the mundane matters.

**SHOT:** Show the paper link on screen.

> If you want to dig deeper, the paper is "Why do LLMs attend to the first token?" 
> Barbero et al., COLM 2025. And you can play with this notebook live — the link is in 
> the description.

*Fade out or go silent for 2 seconds.*

---

## Recording tips

1. **Pre-warm the model** before you hit record. Run the default prompt once so inference 
   is fast and you don't have dead air during the "experiment" click.
2. **Move the mouse deliberately** — don't jitter. Rest 1–2 seconds on key visuals (the 
   86% number, the heatmap column, the experiment bars).
3. **Let pauses breathe** — a 2-second silence is better than filler words.
4. **Audio** — quiet room, clear voice, not too fast. ~150 words/min is a good pace.
5. **Cut afterward** if you need to — trim any false starts or long pauses.
6. **Subtitles** optional but add polish: turn on auto-captions in your video editor.

---

## Shot checklist (for editing)

- [ ] 0:00 Hero section with title + stats
- [ ] 0:20 Token strip with first token highlighted
- [ ] 0:30 Heatmap with tooltip hover
- [ ] 0:45 Bar chart (attention received)
- [ ] 1:00 Headline percentage stat
- [ ] 1:15 Text explanation (section 04)
- [ ] 1:50 Layer-wise sink curve + slider drag
- [ ] 2:20 Experiment button
- [ ] 2:30 Experiment bar chart
- [ ] 2:50 (Optional) Over-mixing meter
- [ ] 3:10 (Optional) Model-size comparison
- [ ] 3:15 "So what?" closing text
- [ ] 3:30 Paper link visible

---

## Upload & share

- **Format:** MP4, H.264, 1080p+ (YouTube/molab accept this).
- **Length:** 2:30–3:30 (YouTube's default recommendation is 2–4 min for discovery).
- **Upload to:** YouTube (unlisted link for the competition form) or Vimeo (if you prefer).
- **Title & description:** Include the molab link, the paper (arXiv:2504.02732), and 
  your GitHub repo.
