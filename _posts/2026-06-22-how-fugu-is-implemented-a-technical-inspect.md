---
layout: post
title: "How Fugu Is Implemented — A Technical Inspection"
date: 2026-06-22T00:00:00.000+08:00
categories: [Blog]
tags: [fugu, llm-orchestration, agentic-ai, systems]
math: true
excerpt: "A technical inspection of Sakana Fugu as a learned orchestrator: a small routing policy over frozen frontier models, adapted with SVF, a linear head, and evolutionary refinement."
---

## 0. What Sakana Fugu is

Sakana Fugu looks like a single OpenAI-compatible model, but architecturally it is a **learned orchestrator**. You call one endpoint; behind it, Fugu routes the query across a pool of frontier LLMs from different vendors (Gemini-3.1-Pro, Claude-Opus-4.8, GPT-5.5, and others) and returns one answer. The product promise is "frontier capability without single-vendor dependency": because the worker pool is swappable and no provider is hardcoded into the architecture, Fugu can absorb availability and policy constraints while still reaching — and on several benchmarks exceeding — the best single model. It ships in two modes: **Fugu** (latency-optimized, one worker per turn) and **Fugu-Ultra** (quality-optimized, a planned multi-step workflow). The public `SakanaAI/fugu` GitHub repo is only an installer for the closed API; the method itself is documented in the Sakana Fugu technical report and two papers, **TRINITY** and **Conductor** ([Sakana AI, 2026](#ref-sakana-fugu-report); [Xu et al., 2026](#ref-trinity); [Nielsen et al., 2026](#ref-conductor)).

The key reframing is simple: **Fugu is not a model; it is a policy over models.** Sakana did not train a better frontier LLM. It trained a compact decision-maker that learns *which existing LLM to ask, for what, and in what order*, while the expensive language generation stays inside frozen worker models.

<style>
.fugu-glance {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.85rem;
  margin: 1.6rem 0 1.9rem;
}
.fugu-glance div {
  border: 1px solid #d8dee9;
  border-left: 4px solid #2563eb;
  border-radius: 8px;
  padding: 0.85rem 0.95rem;
  background: #f8fafc;
}
.fugu-glance strong {
  display: block;
  margin-bottom: 0.3rem;
  color: #111827;
}
.fugu-glance span {
  display: block;
  color: #475569;
  font-size: 0.92rem;
  line-height: 1.45;
}
</style>

<div class="fugu-glance" aria-label="Fugu implementation overview">
  <div><strong>Fugu</strong><span>Low-latency routing: one hidden state, one linear head, one selected worker.</span></div>
  <div><strong>Fugu-Ultra</strong><span>Workflow planning: one Conductor plan, several worker calls, explicit data flow.</span></div>
  <div><strong>Tiny learned surface</strong><span>~19.5K trainable numbers on top of a frozen small backbone and frozen frontier workers.</span></div>
  <div><strong>Executable evidence</strong><span>The released TRINITY checkpoint reproduces the routing fixture at 95% joint accuracy.</span></div>
</div>

## 0.1 How the mechanism works

Under the product surface, the mechanism is small. Fugu runs a **~0.6B-parameter backbone (Qwen3-0.6B) that never produces the user-facing answer**. For a conversation state, it computes one hidden vector and feeds that vector to a **bias-free linear head** that scores the worker models. The highest-scoring worker receives the query, and that worker's reply is what the user sees. The orchestrator's own text generation is discarded; only its routing logits matter. A routing decision therefore costs roughly one backbone forward pass plus one matrix multiply, not a full autoregressive decode. That is the latency story ([Sakana AI, 2026](#ref-sakana-fugu-report); [Xu et al., 2026](#ref-trinity)).

What makes it *learned* rather than hand-tuned is the trainable surface: the routing head plus a small set of **singular-value scale offsets** on the backbone, together about 19.5K trainable numbers (§2.3). These parameters are optimized so the routing logits correlate with which worker actually succeeds. Training is gradient-free — an evolutionary strategy over end-to-end task success — because the signal is sparse, expensive, and weakly coupled. Empirically, the learned routing tracks model strengths: coding-terminal work moves toward GPT-style workers, graduate science toward Gemini-style workers, and a 37-case routing fixture is reproduced at 95–100% (§6).

**Fugu**, the low-latency variant, stops there: read the hidden state, pick one worker, optionally assign a Thinker/Verifier role, then repeat until a verifier accepts or the turn budget is exhausted. **Fugu-Ultra** replaces this per-turn picker with a larger 7B "Conductor" that writes a whole **workflow graph** in one shot: a list of subtasks, the worker assigned to each, and the earlier outputs each worker may see. The graph is then executed with intra-workflow isolation and shared tool-call memory across the conversation (§5). In both variants, **no worker weights are modified**; Fugu is composition over frozen external models ([Nielsen et al., 2026](#ref-conductor)).

<figure class="graf graf--figure">
<img src="/images/blog/fugu-implementation/routing-loop.svg" alt="Diagram of Fugu taking a raw transcript, running a Qwen3-0.6B forward pass, applying a linear routing head, selecting one frontier worker, and returning the worker answer." width="1200" height="650" loading="eager" decoding="async" fetchpriority="high">
<figcaption>Fugu's low-latency path is a routing decision, not user-facing generation by the orchestrator.</figcaption>
</figure>

---

## 1. The core idea

§0 gave the shape. The technical bet is even narrower: the orchestrator commits to a single structural hypothesis.

> **the penultimate-token hidden state $h \in \mathbb{R}^d$ of a mostly frozen
> LM already linearly encodes which large model should act next.**

If that is true, no generation, fine-tuned judge, or learned attention over candidates is needed. A single matrix multiplication, $W_{\text{head}}h$, is enough to route. The rest of the design follows from that hypothesis: a linear head (§2.1), a cheap way to adjust the backbone representation without full fine-tuning (SVF, §2.2), a ~19.5K-parameter training surface small enough for gradient-free optimization (§2.3, §3), and the per-turn or workflow loops that consume the routing decision (§4, §5). The hypothesis is also testable: on a 37-case fixture, the linear head reproduces reference routing at 95–100% (§6).

---

## 2. Fugu — parametrization

### 2.1 The selection policy

**Variant boundary.** The executable evidence below comes from the **academic TRINITY checkpoint** (`model_iter_60.npy`), whose head emits $L+3 = 10$ logits. The **production Fugu** described in the report drops roles and emits only $L$ logits: same backbone, narrower head. Claims about the role-emitting head are TRINITY claims; claims about role-free routing are product Fugu claims.

Let $h(s) \in \mathbb{R}^{d}$ be the backbone hidden state at the penultimate token for transcript state $s$ ($d = 1024$ for Qwen3-0.6B). A bias-free linear head $f_\theta : \mathbb{R}^d \to \mathbb{R}^{L}$ induces a categorical policy over workers:

$$
\pi_\theta(i \mid s) \;=\; \frac{\exp\!\big(f_\theta(h(s))_i\big)}{\sum_{i'=1}^{L}\exp\!\big(f_\theta(h(s))_{i'}\big)},
\qquad i \in \{1,\dots,L\}.
$$

The head weight is $W_{\text{head}} \in \mathbb{R}^{L \times d}$ (or $\mathbb{R}^{(L+3)\times d}$ for the academic variant), so $f_\theta(h) = W_{\text{head}}\, h$. The orchestrator's *generated text is discarded*. Only the logits matter, which is why $h$ can be read without running a user-facing autoregressive decode.

In the **academic TRINITY** variant the head emits $L+3$ logits; the last three select a role $\rho \in \{\text{Worker}, \text{Thinker}, \text{Verifier}\}$ from an independent softmax over those coordinates. Writing $\ell = W_{\text{head}}\, h \in \mathbb{R}^{L+3}$, agent and role are drawn (or, at eval, taken as argmax) from the two sub-vectors:

$$
a \sim \mathrm{softmax}(\ell_{1:L}), \qquad
\rho \sim \mathrm{softmax}(\ell_{L+1:L+3}).
$$

The **production Fugu** drops roles entirely ($L$ logits only), shrinking the coordination space to pure model selection for latency.

### 2.2 Singular-value fine-tuning (SVF) of the backbone

To adapt the backbone's *representation* without full fine-tuning, Fugu uses SVF, following the singular-value adaptation line used in Transformer-Squared ([Sun et al., 2025](#ref-transformer-squared)). For each selected weight matrix $W \in \mathbb{R}^{m \times n}$, take its singular value decomposition

$$
W = U \Sigma V^\top, \qquad
\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_r), \quad r = \min(m,n),
$$

and **freeze the orthogonal factors $U, V$**, learning only a per-singular-value scale. With a trainable offset vector $\delta \in \mathbb{R}^{r}$, the adapted matrix is reconstructed as

$$
\tilde\sigma_k = (1 + \delta_k)\,\sigma_k, \qquad
\widetilde{W} \;=\; \big(U\, \mathrm{diag}(\tilde\sigma)\, V^\top\big)\cdot
\underbrace{\frac{\sum_k \sigma_k}{\sum_k \tilde\sigma_k}}_{\text{energy preservation}}.
$$

The trailing scalar renormalizes total spectral energy so activation scale does not drift. $\delta = 0$ recovers $W$ exactly.

### 2.3 The trainable parameter vector

All learnable parameters concatenate into a single flat vector $x \in \mathbb{R}^{N}$, $N = 19456$, verified exactly from the released checkpoint:

$$
x = \big[\; \underbrace{\delta^{(1)}, \dots, \delta^{(9)}}_{\text{SVF offsets, } 9216}
\;\big\Vert\;
\underbrace{\mathrm{vec}(W_{\text{head}})}_{\text{head, } 10240}\;\big],
\qquad
9216 + 10240 = 19456.
$$

SVF spans **9 matrices**, each contributing $r = 1024$ offsets, consumed in PyTorch `state_dict` order:

$$
\underbrace{\text{embed\_tokens}}_{1024}
,\;
\underbrace{\{q,k,v,o,\text{gate},\text{up},\text{down}\}\text{ of layer }26}_{7 \times 1024}
,\;
\underbrace{\text{lm\_head}}_{1024}
\;=\; 9216.
$$

The head block reshapes as $W_{\text{head}} = \mathrm{reshape}(x_{9216:},\, (L{+}3,\, d)) = (10, 1024)$ for $L = 7$.

> Total learnable parameters $\approx 19.5\text{K}$ — orders of magnitude below
> any fine-tune of the 0.6B backbone, and the reason an evolutionary strategy
> is tractable (§3.2). 

<figure class="graf graf--figure">
<img src="/images/blog/fugu-implementation/trainable-surface.svg" alt="Diagram showing the 19,456 trainable Fugu parameters split into 9,216 SVF singular-value offsets and 10,240 routing-head weights." width="1200" height="680" loading="lazy" decoding="async">
<figcaption>The checkpoint is small enough to inspect directly: 9,216 singular-value offsets plus a 10-by-1024 routing head.</figcaption>
</figure>

---

## 3. Fugu — training

Training is **two-stage** in the product report: a supervised warm start on single-step tasks, then an evolutionary refinement on end-to-end trajectories ([Sakana AI, 2026](#ref-sakana-fugu-report)). The open academic TRINITY artifact covers the evolved-coordinator line but does not expose the product SFT stage ([Xu et al., 2026](#ref-trinity)).

### 3.1 Stage 1 — supervised fine-tuning (SFT) on single-step tasks

For each question $q$ in a verifiable dataset $\mathcal{D}$, every worker $M_i$ is run $R$ times and scored against the ground truth, giving a mean reward and a performance vector over the pool:

$$
\bar r_{q,i} = \frac{1}{R}\sum_{j=1}^{R} r^{M_i}_{q,j},
\qquad
s_q = (\bar r_{q,1}, \dots, \bar r_{q,L}).
$$

Rather than supervising on the hard $\arg\max$ worker, the scores become a **soft target** through a temperature softmax, preserving reward magnitudes:

$$
p_q(i) = \frac{\exp(\bar r_{q,i}/\tau)}{\sum_{i'=1}^{L}\exp(\bar r_{q,i'}/\tau)}.
$$

Both the head and the SVF scales are trained to minimize the forward KL of the soft target $p_q$ relative to the policy (target first argument), i.e. $D_{\mathrm{KL}}(p_q \,\Vert\, \pi_\theta)$:

$$
\mathcal{L}_{\text{SFT}}(\theta)
= \frac{1}{|\mathcal{D}|}\sum_{q \in \mathcal{D}}
D_{\mathrm{KL}}\!\big(p_q(\cdot)\,\Vert\,\pi_\theta(\cdot \mid q)\big).
$$

This stage places $\theta$ in a strong basin and is the *only* place gradients are used. The temperature $\tau$ is not disclosed. This stage exists **only in production Fugu**; the open TRINITY submission has no SFT stage.

### 3.2 Stage 2 — sep-CMA-ES on end-to-end trajectories

The second stage optimizes the **expected terminal reward** of the orchestrator over multi-turn agentic rollouts $\tau$ drawn from real coding-assistant environments such as Claude Code, Codex, and OpenCode. The terminal reward $R(\tau) \in \{0,1\}$ records whether the task completed:

$$
J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big].
$$

This objective is optimized **without gradients**. The parameters are weakly coupled, each evaluation is expensive because every rollout contains many LLM calls, and each terminal result is a Bernoulli draw. REINFORCE-style gradients would be extremely noisy. The paper instead uses **separable CMA-ES**: a CMA-ES variant whose covariance is constrained to be diagonal, $C = \mathrm{diag}(c_1^2,\dots,c_N^2)$, reducing storage and update cost from quadratic/cubic in $N$ to linear in $N$ ([Ros and Hansen, 2008](#ref-ros-hansen-sepcma)). In pycma, the diagonal adaptation is carried by a per-coordinate step-size vector `sigma_vec` rather than by a literal diagonal $C$ matrix; §7 returns to that implementation detail.

Maintaining a parent $m \in \mathbb{R}^N$, a scalar step size $\sigma$, and a per-coordinate scale $d \in \mathbb{R}^N$ (the diagonal), each iteration samples a population of $\lambda$ candidates:

$$
x_i = m + \sigma\, d \odot z_i, \qquad z_i \sim \mathcal{N}(0, I), \qquad i = 1, \dots, \lambda.
$$

Each candidate's fitness estimates $J$ by averaging terminal reward over $R_{\text{rep}}$ replicated rollouts, then adding the shaping terms reported in the released log. The exact normalization of the turn-count term is not exposed:
$$
\hat J(x_i) = \underbrace{\frac{1}{R_{\text{rep}}}\sum_{e=1}^{R_{\text{rep}}} R(\tau_e)}_{\text{mean accuracy}}
\;+\; w_{\text{div}}\, H(\text{agents})
\;+\; w_{\text{turn}}\, \bar T
\;-\; w_{\text{cost}}\, \bar C,
$$

where $H$ is the entropy of the agent-selection distribution (diversity bonus) and $\bar T, \bar C$ are mean turn count and cost. The top-$\mu$ candidates by fitness are recombined by rank-weighted averaging to form the next parent:

$$
m' = m + \sum_{k=1}^{\mu} w_k\, (x_{k:\lambda} - m) \;=\; \sum_{k=1}^{\mu} w_k\, x_{k:\lambda},
\qquad \sum_k w_k = 1,
$$

with $x_{k:\lambda}$ the $k$-th best candidate under the fitness ranking (the step size $\sigma$ enters only the sampling above, not this mean update).

**Real configuration** (from the released training log `es_log.json`):

$$
\begin{aligned}
&\text{iters} = 60,\quad \sigma_0 = 0.03,\quad R_{\text{rep}} = 16,\quad \text{seed} = 42, \\
&w_{\text{div}} = 0.15,\quad w_{\text{turn}} = 0.10,\quad w_{\text{cost}} = 0, \\
&L = 7,\quad \text{max\_turns} = 5,\quad \text{SVF layer} = 26.
\end{aligned}
$$

With $\texttt{popsize\_override}=0$, pycma defaults apply, giving the population and parent counts in closed form for $N = 19456$:
$$
\lambda = 4 + \lfloor 3 \ln N \rfloor = 33, \qquad
\mu = \lfloor \lambda/2 \rfloor = 16, \qquad
\mu_{\text{eff}} \approx 9.44,
$$

with log-rank recombination weights $w_k \propto \ln(\mu + \tfrac12) - \ln k$ ($w_1 \approx 0.193$, normalized to sum 1). The paper's "$\lambda \approx 32$" is exactly this rounding. The released checkpoint `model_iter_60.npy` is the $m$ at iteration 60.

> The ask/tell loop itself is **not in the shipped code** (it lives in an
> un-released `experiments/with_training/` module), so the structure above is
> reconstructed from the trainer signature, the eval-path job submission, and
> pycma defaults. The separable choice is **load-bearing for feasibility**: a
> full $N\times N$ covariance is intractable at $N=19456$ — but within 60
> iterations even the diagonal scale $d$ barely moves; only the scalar $\sigma$
> changes materially, so the run behaves like an isotropic ES (dissected in §7).
> 

---

## 4. Fugu — the inference loop

At serving time, the orchestrator runs a bounded multi-turn loop over the transcript $\mathcal{C}$. With SVF applied to the backbone, each turn $t$:

1. Format the transcript as **raw** `"role: content\n"` text (not a chat
template — proven decisive in §6), tokenize, forward pass.
2. Extract $h_t$ as the hidden state at position $-2$.
3. Route: $\ell_t = W_{\text{head}}\, h_t$; sample/argmax agent $a_t$ and
(TRINITY) role $\rho_t$.
4. Inject the role's system prompt, dispatch to worker $M_{a_t}$, append its
reply to $\mathcal{C}$.

The loop terminates at the stopping time

$$
\tau = \min\big\{\, t \le K : \rho_t = \text{Verifier} \;\wedge\; u_t = \texttt{ACCEPT} \,\big\},
\qquad K = 5,
$$

returning $O_\tau$ (the latest worker response); if no verifier accepts, it returns the last response at $t = K$. A Thinker turn may emit `<suggested_role>` which **overrides** the head's role choice on the next turn.

---

## 5. Fugu-Ultra — the Conductor line

Fugu-Ultra replaces the per-turn selector with a 7B LM that emits an **entire agentic workflow** in natural language. That workflow is parsed into three equal-length lists, which together define a DAG over workers ([Nielsen et al., 2026](#ref-conductor)):

$$
\texttt{model\_id}[1{:}T],\quad
\texttt{subtasks}[1{:}T],\quad
\texttt{access\_list}[1{:}T],
$$

Step $t$ dispatches `subtasks[t]` to worker `model_id[t]`; `access_list[t]` indexes which earlier outputs enter that worker's context. The visibility relation must be topologically ordered — forward references fail — so the workflow is a DAG executed sequentially:

$$
\text{ctx}(t) = \big\{\,(\,\texttt{subtask}_j,\, o_j\,) : j \in \texttt{access\_list}[t]\,\big\},
\qquad j < t .
$$

### 5.1 Training — GRPO

The Conductor is trained with GRPO on a progressive reward. The $0/0.5/1$ structure is the documented design; in code, the correctness term is supplied by an external task-specific function rather than a hardcoded constant. The GRPO form follows DeepSeekMath's group-relative policy optimization, with the advantage normalized over grouped completions in the standard policy-gradient framing ([Shao et al., 2024](#ref-deepseekmath); [Sutton et al., 1999](#ref-policy-gradient)):

$$
r_i =
\begin{cases}
0 & \text{lists unparseable (format fail)} \\
0.5 & \text{parsed, executed, but wrong} \\
1 & \text{parsed, executed, correct,}
\end{cases}
$$

maximizing, over a group of $G$ sampled workflows per query, the clipped surrogate with group-normalized advantage and **no KL penalty in production** ($\beta = 0$):

$$
J(\theta) = \mathbb{E}_{q,\{o_i\}\sim\pi_\theta(\cdot\mid q)}
\!\left[\frac{1}{G}\sum_{i=1}^{G}
\min\!\big(\rho_i A_i,\; \mathrm{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\,A_i\big)
- \beta\, D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})\right],
$$

$$
\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},
\qquad
A_i = \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})}.
$$

A recursion finetune lets the Conductor name **itself** as a worker, with a discount $\gamma = 0.2$ on the non-recursive round and per-round reward normalization.

### 5.2 Production-only orchestration memory

Two mechanisms, described in the report but not in the papers, make multi-agent function calling coherent:

- **Intra-workflow isolation** — within one workflow, agent $a$ sees agent $b$'s
trajectory *only* through `access_list`; otherwise each keeps its own transcript. This prevents "orchestration collapse," where the first agent's path steers all others.
- **Inter-workflow shared memory** — across the multi-turn conversation, agents
share tool-call history, avoiding redundant re-discovery.

Formally, the context visible to agent $a$ at workflow $W$, turn $t$:

$$
\text{ctx}_a(W,t) = \underbrace{\text{own-history}_a}_{\text{isolated}}
\;\cup\; \underbrace{\{o_j : j \in \texttt{access\_list}\}}_{\text{topology}}
\;\cup\; \underbrace{\text{tool-memory}(\,<W\,)}_{\text{shared across workflows}}.
$$

<figure class="graf graf--figure">
<img src="/images/blog/fugu-implementation/ultra-dag.svg" alt="Diagram of Fugu-Ultra's Conductor emitting model_id, subtasks, and access_list arrays that execute as a topologically ordered workflow over worker models." width="1200" height="700" loading="lazy" decoding="async">
<figcaption>Fugu-Ultra differs from Fugu by planning the data flow explicitly: workers see earlier outputs only when the Conductor's access list grants them.</figcaption>
</figure>

---

## 6. Execution proof

The Fugu (TRINITY) inference path was reproduced end-to-end: the released `model_iter_60.npy`, Qwen3-0.6B on an A800, SVF applied in `state_dict` order, and a 37-case fixture with expected $(a, \rho)$ labels.

| Input formatting                  | agent acc. | role acc. | joint   |
| --------------------------------- | ---------- | --------- | ------- |
| naive baseline (guess mode class) | 51%        | 49%       | —       |
| **raw `"role: content"`**         | **95%**    | **100%**  | **95%** |
| chat template                     | 41%        | 11%       | 5%      |

Against the best constant-guess null (always picking the modal class, $p_0 = 19/37$ for agent and $18/37$ for role), a one-sided binomial test puts the agent result at $P(\ge 35/37) \approx 1.2\times10^{-8}$ and the role result at $P(=37/37) \approx 2.6\times10^{-12}$. This is not a chance fit. It simultaneously confirms the $9216 \,\Vert\, 10240$ checkpoint split and ordering, the 9-matrix SVF layout, the energy-preserving reconstruction, the raw-transcript input convention, and the pairing between checkpoint and inference path. The two agent misses are a sub-threshold logit margin ($0.214 < 0.24$, fp32 noise) and a unicode-emoji edge case.

---

## 7. What remains uncertain — stated plainly

A few pieces are not directly confirmed, and it is worth separating them from the parts above.

The most important ambiguity is what "sep" in sep-CMA-ES contributes in this run. It is not Sakana's invention; it is the standard separable variant of Ros and Hansen, which restricts covariance to a diagonal and cuts storage/eigendecomposition cost from $O(N^2)$/$O(N^3)$ to $O(N)$ ([Ros and Hansen, 2008](#ref-ros-hansen-sepcma)). In pycma v4.4.4, `CMA_diagonal=True` does **not** maintain a literal diagonal $C$ matrix. It freezes the sampler's $C$ at $I$ (`GaussStandardConstant`, "no update") and moves per-coordinate adaptation into `sigma_vec`, so candidates are sampled as $x_i = m + \sigma\cdot\texttt{sigma\_vec}\odot z_i$ with $z_i\sim\mathcal N(0,I)$. The effect is diagonal covariance, implemented outside the sampler.

At Fugu's scale, this distinction matters. Full CMA is **computationally infeasible** at $N=19456$ because it would require an $N\times N$ matrix and eigendecomposition. The separable form is therefore a feasibility condition, not a cosmetic variant. But within 60 iterations, even sep-CMA's `sigma_vec` barely diverges (std $\sim\!6\times10^{-4}$). The only quantity that moves materially is the scalar step size $\sigma$ ($0.03\to0.002$). In practice, the run behaves like an isotropic $(\mu/\mu_w,\lambda)$-ES: scalar step-size control plus fitness-weighted mean recombination, with the diagonal machinery present but barely exercised at this budget. One correction follows from that: near-zero cross-correlation across SVF blocks is not evidence for a diagonal $C$; it is what any mode would show when covariance adaptation barely moves.

The training **ask/tell loop** is not in the shipped code; it lives in an unreleased `experiments/with_training/` module. The reconstruction here is mostly pinned by the trainer signature, eval-path job submission, released config, and pycma defaults. The remaining details are inference rather than directly published artifact.

Everything else that remains hidden is either a **credential** (worker API keys, unrecoverable by design) or a **tuning magnitude** (the SFT temperature $\tau$, production GRPO $\beta/\epsilon$ and batch sizes, or the live worker pool and routing weights that rotate over time). Those are closed knobs, not missing architecture. The structure itself is clear: a tiny coordinator over frozen workers, an SVF-adapted backbone, a linear selection head, and evolutionary refinement on end-to-end task success.

---

## 8. One-paragraph summary

Fugu attaches a $\sim$19.5K-parameter trainable surface to a frozen Qwen3-0.6B: a bias-free linear head $W_{\text{head}} \in \mathbb{R}^{(L+3)\times d}$ plus SVF singular-value offsets on 9 backbone matrices. The penultimate-token hidden state is mapped to worker (and, in TRINITY, role) logits; the selected frontier LLM does the actual work, and the orchestrator never decodes user-facing text. Production Fugu is warm-started by KL-matching a temperature-softmax of measured per-worker rewards, then refined by separable CMA-ES directly on end-to-end task success $J(\theta)=\mathbb{E}_\tau[R(\tau)]$ — although at 19456-D over 60 iterations, that behaves in practice like an isotropic step-size-adapted ES. Fugu-Ultra swaps the per-turn selector for a GRPO-trained 7B Conductor that emits a full topological-DAG workflow with isolated-but-shared agent memory. No worker weights are modified. The system is macro-level composition over heterogeneous, swappable APIs.

## [References]

- <a id="ref-sakana-fugu-report"></a>Sakana AI. 2026. "Sakana Fugu." Technical report. <https://sakana.ai/fugu>.
- <a id="ref-trinity"></a>Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, and Yujin Tang. 2026. "TRINITY: An Evolved LLM Coordinator." arXiv:2512.04695. <https://arxiv.org/abs/2512.04695>.
- <a id="ref-conductor"></a>Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. 2026. "Learning to Orchestrate Agents in Natural Language with the Conductor." arXiv:2512.04388. <https://arxiv.org/abs/2512.04388>.
- <a id="ref-transformer-squared"></a>Qi Sun, Edoardo Cetin, and Yujin Tang. 2025. "Transformer-Squared: Self-Adaptive LLMs." ICLR 2025. <https://openreview.net/forum?id=dh4t9qmcvK>.
- <a id="ref-ros-hansen-sepcma"></a>Raymond Ros and Nikolaus Hansen. 2008. "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity." PPSN X. <https://doi.org/10.1007/978-3-540-87700-4_30>.
- <a id="ref-deepseekmath"></a>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, et al. 2024. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. <https://arxiv.org/abs/2402.03300>.
- <a id="ref-policy-gradient"></a>Richard S. Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." NeurIPS 1999.
