---
layout: post
title: "What Is RLCD? The Secret Behind Jev"
date: 2026-09-21T00:00:00.000+08:00
permalink: /blog/what-is-rlcd-the-secret-behind-jev/
categories: [Blog]
tags: [jev, rlcd, reward-modeling, plackett-luce, calibration, brier-score]
math: true
toc_heading_level: 2
image: "/images/blog/what-is-rlcd-the-secret-behind-jev/01-rlcd-lineage.png"
excerpt: "RLCD is a calibrated, schema-conditioned extension of pairwise reward modeling: Bradley–Terry becomes Plackett–Luce, and the reward model becomes Jev's typed decision interface."
---

*From pairwise reward modeling to calibrated, multiway decisions*

Jev looks mysterious when viewed as an alternative to a language model. It becomes much simpler when viewed as the next step in reward modeling.

The core idea is:

<div class="math-display" markdown="0">
\[
\text{RLCD}
=
\text{multiway preference modeling}
+
\text{probability calibration}
\]
</div>

More specifically, RLCD is a schema-conditioned Plackett–Luce objective. Jev turns that objective into a product by adding typed outputs and parallel inference.

That is the secret: the reward model is no longer hidden behind a generator. The reward model becomes the model.

<figure id="figure-rlcd-lineage" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/01-rlcd-lineage.svg" alt="Four-stage diagram showing scalar reward becoming pairwise preference, multiway choice, and finally a calibrated decision served through the Jev API." width="1600" height="900" loading="eager" fetchpriority="high" decoding="async">
<figcaption>Figure 1. The learned object changes at each step: a scalar reward becomes a preference, the preference becomes a multiway distribution, and calibration turns that distribution into a decision interface.</figcaption>
</figure>

## Reward Modeling Started with a Scalar

A conventional reward model receives a context <span class="math-inline" markdown="0">\(x\)</span> and a candidate answer <span class="math-inline" markdown="0">\(a\)</span>, then produces a scalar:

<div class="math-display" markdown="0">
\[
r_\theta(x,a)\in\mathbb{R}
\]
</div>

Outcome reward models score the final answer. Process reward models score individual reasoning steps. In both cases, the learned object is an absolute-looking number.

The problem is that this number is not actually absolute.

A reward of <span class="math-inline" markdown="0">\(0.8\)</span> does not have a stable meaning across problems, candidate pools, checkpoints, or model families. It is mainly useful for comparing candidates generated under similar conditions:

<div class="math-display" markdown="0">
\[
r_\theta(x,a_1) &gt; r_\theta(x,a_2)
\]
</div>

The operational signal was always relative preference. The scalar merely hid it.

## PPRM Made the Preference Explicit

LLaMA-Berry’s Pairwise Preference Reward Model, or PPRM, exposes the comparison directly.

Given a problem <span class="math-inline" markdown="0">\(x\)</span> and two solutions <span class="math-inline" markdown="0">\(a_1\)</span> and <span class="math-inline" markdown="0">\(a_2\)</span>, PPRM answers:

> Is the first answer better than the second answer?

Its probability has the form:

<div class="math-display" markdown="0">
\[
P(a_1 \succ a_2\mid x)
=
\frac{\exp u_\theta(x,a_1)}
{\exp u_\theta(x,a_1)+\exp u_\theta(x,a_2)}
\]
</div>

Equivalently:

<div class="math-display" markdown="0">
\[
P(a_1 \succ a_2\mid x)
=
\sigma\left(
u_\theta(x,a_1)-u_\theta(x,a_2)
\right)
\]
</div>

This is the Bradley–Terry model.

LLaMA-Berry implements the comparison as a constrained language-model decision over `Yes` and `No` tokens. It trains the evaluator on almost 7.8 million mathematical-solution pairs and uses DPO to improve the pairwise prediction task. The essential change is conceptual: reward modeling becomes preference-probability modeling. See [the LLaMA-Berry paper](https://aclanthology.org/2025.naacl-long.375.pdf).

PPRM still contains a latent scalar utility <span class="math-inline" markdown="0">\(u_\theta(x,a)\)</span>, but that utility is no longer presented as an absolute reward. It becomes meaningful through a normalized comparison.

LLaMA-Berry subsequently uses Enhanced Borda Count to aggregate pairwise comparisons inside MCTS. That is downstream search machinery. EBC neither defines PPRM’s preference loss nor provides the bridge from PPRM to RLCD.

The relevant lineage is simply:

<div class="math-display" markdown="0">
\[
\text{scalar reward}
\rightarrow
\text{pairwise preference}
\rightarrow
\text{multiway preference}
\rightarrow
\text{calibrated decision}
\]
</div>

## Plackett–Luce Is the Multiway PPRM

PPRM compares two candidates. A real decision interface usually receives more than two.

Let the candidate set be:

<div class="math-display" markdown="0">
\[
A=\{a_1,a_2,\dots,a_K\}
\]
</div>

Assign each candidate a context-dependent utility:

<div class="math-display" markdown="0">
\[
u_i=u_\theta(x,a_i)
\]
</div>

Then normalize all candidates together:

<div class="math-display" markdown="0">
\[
P(a_i\mid x,A)
=
\frac{\exp u_i}
{\sum_{j=1}^{K}\exp u_j}
\]
</div>

This is the Luce choice model, also known as multinomial logit. It is the top-one form of the Plackett–Luce family.

When <span class="math-inline" markdown="0">\(K=2\)</span>, it reduces exactly to Bradley–Terry:

<div class="math-display" markdown="0">
\[
P(a_1\mid x,\{a_1,a_2\})
=
\frac{\exp u_1}{\exp u_1+\exp u_2}
\]
</div>

PPRM is therefore the binary case of the same choice geometry.

If the supervision contains a complete ranking

<div class="math-display" markdown="0">
\[
a_{\pi_1}\succ a_{\pi_2}\succ\dots\succ a_{\pi_K},
\]
</div>

the full Plackett–Luce likelihood repeatedly selects the next-best remaining candidate:

<div class="math-display" markdown="0">
\[
P(\pi\mid x)
=
\prod_{t=1}^{K}
\frac{\exp u_{\pi_t}}
{\sum_{j=t}^{K}\exp u_{\pi_j}}
\]
</div>

The corresponding loss is:

<div class="math-display" markdown="0">
\[
\mathcal{L}_{\mathrm{PL}}
=
-\sum_{t=1}^{K}
\log
\frac{\exp u_{\pi_t}}
{\sum_{j=t}^{K}\exp u_{\pi_j}}
\]
</div>

When the label specifies only one correct choice <span class="math-inline" markdown="0">\(y\)</span>, the loss becomes:

<div class="math-display" markdown="0">
\[
\mathcal{L}_{\mathrm{choice}}
=
-\log
\frac{\exp u_y}
{\sum_j\exp u_j}
\]
</div>

That is the first stage of the Plackett–Luce likelihood: a multiway extension of PPRM.

This is the mathematical center of RLCD.

<figure id="figure-pairwise-multiway" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/02-pairwise-to-multiway.svg" alt="Side-by-side diagram of Bradley–Terry pairwise preference and Luce multiway choice sharing the same latent-utility normalization." width="1600" height="920" loading="lazy" decoding="async">
<figcaption>Figure 2. Bradley–Terry and PPRM are the two-candidate case of the same Luce choice geometry. Plackett–Luce extends that normalization from one choice to a complete or partial ranking.</figcaption>
</figure>

## RLCD Adds Calibration

Plackett–Luce gives us a probability distribution, but normalization is not calibration.

A softmax vector always sums to one. That does not mean a prediction reported as <span class="math-inline" markdown="0">\(0.8\)</span> is correct 80% of the time.

Calibration adds that empirical meaning:

<div class="math-display" markdown="0">
\[
P(Y=\hat{Y}\mid \hat{P}=p)\approx p
\]
</div>

Across predictions assigned probability <span class="math-inline" markdown="0">\(0.8\)</span>, approximately 80% should be correct. This is also the contract TypeSafe gives for RLCD: Jev returns decisions and probabilities, and higher reported probabilities should correspond to higher observed accuracy. See [TypeSafe’s RLCD primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

A minimal implementation uses a proper scoring rule such as log loss:

<div class="math-display" markdown="0">
\[
\mathcal{L}_{\mathrm{NLL}}=-\log p_y
\]
</div>

### Brier calibration: confidence gets a price

The Brier score makes the calibration objective concrete. For a binary `Noul` decision, let <span class="math-inline" markdown="0">\(p=P(Y=1\mid x)\)</span> and <span class="math-inline" markdown="0">\(y\in\{0,1\}\)</span>. The score is:

<div class="math-display" markdown="0">
\[
\operatorname{BS}(p,y)=(p-y)^2
\]
</div>

If the model reports <span class="math-inline" markdown="0">\(p=0.8\)</span>, it receives a score of <span class="math-inline" markdown="0">\(0.04\)</span> when the event occurs and <span class="math-inline" markdown="0">\(0.64\)</span> when it does not. The confidently wrong forecast costs sixteen times as much as the confidently correct one.

This is why the Brier score fits a decision model. It is a strictly proper scoring rule: in expectation, the model minimizes the score by reporting the true conditional probability instead of gaming the threshold. The score was introduced for probabilistic forecasts by [Glenn Brier](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml); its role as a proper scoring rule is developed by [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437).

For a multiway `Choice`, the score extends to the full probability vector. Using the normalization that makes the two-class case match the binary formula:

<div class="math-display" markdown="0">
\[
\operatorname{BS}(\mathbf{p},y)
=
\frac{1}{2}
\sum_{i=1}^{K}
\left(p_i-\mathbb{1}[i=y]\right)^2
\]
</div>

This matters because top-1 accuracy discards probability quality. Two models can choose the same action while reporting <span class="math-inline" markdown="0">\(0.55\)</span> and <span class="math-inline" markdown="0">\(0.99\)</span>. Once outcomes arrive, Brier score tells us whether that extra confidence was earned.

For binary outcomes, the [Murphy decomposition](https://doi.org/10.1175/1520-0450%281973%29012%3C0595%3AANVPOT%3E2.0.CO%3B2) separates the mean score into three terms:

<div class="math-display" markdown="0">
\[
\operatorname{BS}
=
\operatorname{REL}
-
\operatorname{RES}
+
\operatorname{UNC}
\]
</div>

- Reliability <span class="math-inline" markdown="0">\(\operatorname{REL}\)</span> measures the gap between reported probabilities and observed frequencies. Lower is better.
- Resolution <span class="math-inline" markdown="0">\(\operatorname{RES}\)</span> measures whether the model separates cases with different outcome rates. Higher is better.
- Uncertainty <span class="math-inline" markdown="0">\(\operatorname{UNC}\)</span> is the base-rate difficulty of the evaluation set. It is fixed when models are compared on the same data.

A lower Brier score can therefore come from better calibration, better separation of easy and hard cases, or both. A constant base-rate predictor can be calibrated while having zero resolution; Brier exposes that weakness.

<figure id="figure-brier-calibration" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/03-brier-calibration.svg" alt="Three-panel diagram showing the Brier penalty for a correct and incorrect 0.8 forecast, the reliability-resolution-uncertainty decomposition, and an RLCD calibration loop from logged outcomes to execution policy." width="1600" height="920" loading="lazy" decoding="async">
<figcaption>Figure 3. Brier score prices confidence, decomposes forecast quality, and closes the loop from observed outcomes to an operational decision policy.</figcaption>
</figure>

An RLCD implementation can use Brier score twice: as a training loss for the probability head and as a held-out objective for post-hoc calibration. With temperature scaling, the calibration parameter can be selected directly on validation outcomes:

<div class="math-display" markdown="0">
\[
T^*
=
\arg\min_{T&gt;0}
\sum_{n=1}^{N}
\operatorname{BS}\!\left(\mathbf{p}^{(T)}(x_n),y_n\right)
\]
</div>

Temperature scaling then adjusts the sharpness of the distribution:

<div class="math-display" markdown="0">
\[
p_i
=
\frac{\exp(u_i/T)}
{\sum_j\exp(u_j/T)}
\]
</div>

Here <span class="math-inline" markdown="0">\(T\)</span> controls how concentrated the probabilities are without changing their ordering. Brier is the objective; temperature scaling is the calibrator. One measures probability quality, while the other changes the distribution.

This separates two objectives that ordinary reward modeling often conflates:

- Ranking asks whether the best candidate appears first.
- Calibration asks whether the model knows how often that decision is right.

Automation needs both. Ranking selects an action; calibration determines whether software should execute it, defer it, or escalate it.

The useful abstraction is:

<div class="math-display" markdown="0">
\[
\text{RLCD}
=
\text{Plackett–Luce preference loss}
+
\text{calibration constraint}
\]
</div>

<figure id="figure-calibration-control" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/03-calibration-control-signal.svg" alt="Conceptual reliability diagram followed by a decision policy that gathers context, escalates, or executes according to calibrated confidence." width="1600" height="920" loading="lazy" decoding="async">
<figcaption>Figure 4. Calibration attaches empirical meaning to confidence, allowing application-specific policies to decide when to gather context, escalate, or execute. The reliability curve is conceptual, not a Jev benchmark.</figcaption>
</figure>

## Jev Turns the Reward Model into the Product

In the conventional RLHF stack, the reward model is an internal component:

<div class="math-display" markdown="0">
\[
\text{prompt}
\rightarrow
\text{generator}
\rightarrow
\text{candidate response}
\rightarrow
\text{reward model}
\]
</div>

Users interact with the generator. The reward model only trains or evaluates it.

Jev reverses that architecture:

<div class="math-display" markdown="0">
\[
\text{state}
+
\text{candidate schema}
\rightarrow
\text{calibrated decision distribution}
\]
</div>

There is no need to generate an explanation and parse it back into an action. The evaluator itself becomes the runtime interface.

Jev exposes three primitives:

| Jev primitive | Preference-model interpretation |
|---|---|
| `Noul` | Binary Bradley–Terry decision between true and false |
| `Choice` | Luce distribution over <span class="math-inline" markdown="0">\(K\)</span> unordered alternatives |
| `Score` | Distribution over an ordered set of levels |

A `Choice` returns the selected option, the complete probability distribution, and a confidence value. A `Score` returns a position along user-defined levels together with the distribution across those levels. A `Noul` returns the probability that a proposition is true. See [Jev’s primitive documentation](https://docs.typesafe.ai/primitives).

These are not three unrelated capabilities. They are three schemas over the same underlying object:

<div class="math-display" markdown="0">
\[
P(\text{typed outcome}\mid \text{state},\text{question},\text{candidate set})
\]
</div>

Jev is therefore a reward model generalized from “Which answer is better?” to “Which typed outcome should the program select?”

<figure id="figure-reward-model-product" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/04-reward-model-as-product.svg" alt="Architecture comparison showing a conventional RLHF reward model behind a text generator and Jev serving the evaluator directly as typed Noul, Choice, and Score outputs." width="1600" height="940" loading="lazy" decoding="async">
<figcaption>Figure 5. Conventional stacks use the reward model behind the generator. Jev serves the evaluator itself: state and schema in, typed probability distributions out.</figcaption>
</figure>

## Why Jev Can Run in Parallel

Strip away the branding: Jev's **parallel sampler is sequence packing plus an attention mask**, followed by typed decision heads. This is the serving trick behind the speed claim.

Autoregressive language models represent an answer as a token sequence:

<div class="math-display" markdown="0">
\[
P(y\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid x,y_{&lt;t})
\]
</div>

Every token depends on the previous tokens. Latency grows with output length.

A decision model already knows its output space. It only needs to estimate utilities and normalize them:

<div class="math-display" markdown="0">
\[
x,A
\rightarrow
(u_1,\dots,u_K)
\rightarrow
(p_1,\dots,p_K)
\]
</div>

No sentence has to be decoded.

Now pack the shared state, questions, and candidate branches into one sequence:

<div class="math-display" markdown="0">
\[
Z
=
[S;Q_1;C_{1,1};\dots;C_{1,K_1};Q_2;C_{2,1};\dots;C_{m,K_m}]
\]
</div>

The packed sequence is only the physical layout. Its logical layout is a tree:

<div class="math-display" markdown="0">
\[
S
\rightarrow
Q_q
\rightarrow
C_{q,k}
\]
</div>

The attention mask preserves that tree. A question reads the shared state and itself. A candidate reads the shared state, its own question, and its own candidate tokens. It cannot read another question or a sibling candidate. Let <span class="math-inline" markdown="0">\(v(i)\)</span> denote the tree node containing token <span class="math-inline" markdown="0">\(i\)</span>, and let <span class="math-inline" markdown="0">\(v(j)\preceq v(i)\)</span> mean that <span class="math-inline" markdown="0">\(v(j)\)</span> is an ancestor of, or identical to, <span class="math-inline" markdown="0">\(v(i)\)</span>. Then:

<div class="math-display" markdown="0">
\[
M^{\mathrm{tree}}_{ij}
=
\begin{cases}
0, &amp; v(j)\preceq v(i),\\
-\infty, &amp; \text{otherwise}.
\end{cases}
\]
</div>

For a causal backbone, this structural mask is combined with causal order *inside each branch*. Position IDs reset at every branch: all questions start after the same state prefix, and all candidates under a question start after the same state-plus-question prefix. Candidate <span class="math-inline" markdown="0">\(C_{q,2}\)</span> therefore gains no information merely because it was packed after <span class="math-inline" markdown="0">\(C_{q,1}\)</span>.

<div class="math-display" markdown="0">
\[
\operatorname{Attn}(Q,K,V;M)
=
\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt d}+M\right)V
\]
</div>

The result is one accelerator-friendly forward pass that produces every candidate score together. Packing removes repeated prefixes. Tree attention prevents cross-question and cross-candidate contamination. Typed heads normalize those scores into `Noul`, `Choice`, or `Score` probabilities. There is no token-by-token generation loop.

<figure id="figure-parallel-sampler" class="graf graf--figure">
<img src="/images/blog/what-is-rlcd-the-secret-behind-jev/05-parallel-sampler-packing-mask.svg" alt="Tree attention diagram showing a shared state branching into questions and isolated candidates, paired with an attention matrix in which each candidate reads only its ancestors and itself." width="1600" height="980" loading="lazy" decoding="async">
<figcaption>Figure 6. The packed token buffer is logically a tree: state → question → candidate. The mask exposes only a branch's ancestral path, so all candidates can be scored in one forward pass without seeing their siblings.</figcaption>
</figure>

This behavior is exactly the contract in [TypeSafe's documentation](https://docs.typesafe.ai/introduction): questions share the same state, are evaluated independently, and return in parallel. The mechanism itself is established Transformer engineering. Sequence packing with attention masks that prevent cross-contamination was already documented as a general throughput technique in the [sequence-packing literature](https://arxiv.org/abs/2107.02027).

TypeSafe's launch post names a “new model architecture” and a “parallel sampler,” but it publishes no new attention operator, no sampler algorithm, no complexity result, and no ablation that isolates a novel sampling mechanism. A real sampling breakthrough would make those artifacts the center of the announcement. They are absent. What remains is a productized composition of familiar primitives:

<div class="math-display" markdown="0">
\[
\text{parallel sampler}
=
\text{packing}
+
\text{attention mask}
+
\text{typed decision heads}
\]
</div>

For very high-cardinality choices, Jev adds a two-stage procedure: score candidates independently, then make an explicit choice. That is another scheduling decomposition, not a new sampling law. See [TypeSafe's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

The complete system decomposition is therefore:

<div class="math-display" markdown="0">
\[
\text{Jev}
=
\text{RLCD}
+
\text{typed schemas}
+
\text{packing}
+
\text{attention masks}
\]
</div>

RLCD explains what the model learns. Packing and masking explain how the learned decision function is served efficiently. The engineering is useful. It is not a new class of sampler.

## RLCD Is Not a Third Kind of Reward Source

TypeSafe presents RLHF, RLVR, and RLCD as three post-training paths. They are not three mutually exclusive mathematical categories.

RLHF and RLVR primarily describe where the reward comes from:

- RLHF: human preference.
- RLVR: programmatically verifiable outcomes.

RLCD describes what the model is trained to return:

- a constrained decision;
- a probability distribution;
- calibrated uncertainty.

Human comparisons can train RLCD. Verifiable outcomes can train RLCD. Synthetic judges can train RLCD. Logged production outcomes can train RLCD.

The word *reinforcement learning* describes the broader post-training pipeline. The statistical heart of the objective is preference estimation under a proper probabilistic loss. PPO is not required to obtain this structure.

The cleaner taxonomy is:

| Method | Primary training signal | Product output |
|---|---|---|
| RLHF | Human preference | Generated response |
| RLVR | Verifiable reward | Generated reasoning or answer |
| RLCD | Decision outcome and calibration | Typed probability distribution |

RLCD is defined by the output contract, not by a unique source of reward.

## The Thesis Produces Testable Predictions

If Jev is a calibrated, schema-conditioned Plackett–Luce model, its behavior should expose several measurable properties.

### 1. Binary equivalence

A two-option `Choice` and an equivalent `Noul` question should produce closely aligned probabilities:

<div class="math-display" markdown="0">
\[
P(A\mid\{A,B\})
\approx
P(A\succ B)
\]
</div>

### 2. Pairwise–multiway consistency

For two candidates inside a larger set:

<div class="math-display" markdown="0">
\[
\frac{P(a_i\mid A)}{P(a_j\mid A)}
\approx
\exp(u_i-u_j)
\]
</div>

Their relative odds should match a direct pairwise comparison when the context and wording are held constant.

### 3. Candidate-set sensitivity

Vanilla Plackett–Luce satisfies independence of irrelevant alternatives. Adding an unrelated candidate should preserve the odds between existing candidates:

<div class="math-display" markdown="0">
\[
\frac{P(a_i\mid A)}{P(a_j\mid A)}
=
\frac{P(a_i\mid A\cup\{a_k\})}
{P(a_j\mid A\cup\{a_k\})}
\]
</div>

Violations measure how strongly Jev’s utility encoder jointly represents the candidate set.

### 4. Empirical calibration

Predictions can be placed into probability bins. For the <span class="math-inline" markdown="0">\(0.8\)</span> bin, observed accuracy should approach <span class="math-inline" markdown="0">\(0.8\)</span>. For `Noul`, report the reliability curve, mean Brier score, and Murphy decomposition together. For `Choice`, report multiclass Brier score and classwise reliability. These views distinguish a useful calibrated model from one that stays safe by predicting the base rate for every case.

### 5. Order symmetry

Permuting the order of candidate definitions should permute the returned probabilities without changing their values. Any systematic position effect reveals schema-order bias.

These tests turn the RLCD interpretation into a falsifiable model of Jev’s behavior.

## Conclusion

Jev is not fundamentally a language model that learned to emit cleaner JSON. It is a preference model promoted into a software interface.

PPRM provides the first step:

<div class="math-display" markdown="0">
\[
\text{absolute reward}
\rightarrow
\text{pairwise preference probability}
\]
</div>

Plackett–Luce provides the multiway extension:

<div class="math-display" markdown="0">
\[
\text{pairwise preference}
\rightarrow
\text{distribution over candidate actions}
\]
</div>

Calibration makes that distribution operational:

<div class="math-display" markdown="0">
\[
\text{choice probability}
\rightarrow
\text{automation threshold}
\]
</div>

Jev packages the result as typed, parallel inference. It is a calibrated multiway reward model served as an API.

The deepest shift is not from one reinforcement-learning algorithm to another. It is from generating an unconstrained answer to estimating a calibrated distribution over actions already defined by software.

Jev is what happens when the reward model stops grading the product and becomes the product.
