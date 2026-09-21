---
layout: post
title: "SparseLeaf: Synaptic Sparse Fine-Tuning"
date: 2026-09-20T00:00:00.000+08:00
permalink: /blog/sparseleaf-synaptic-sparse-fine-tuning/
categories: [Blog]
tags: [sparseleaf, sparse-fine-tuning, lora, optimization, synaptic-plasticity]
math: true
toc_heading_level: 2
image: "/images/blog/sparseleaf-synaptic-sparse-fine-tuning/01-sparsity-and-rank.png"
excerpt: "Low rank is not sparsity. SparseLeaf constrains which connections can change while preserving high-rank updates, turning parameter-efficient fine-tuning into synaptic plasticity."
---

*Let a small number of connections carry a high-rank update.*

Consider a weight matrix with 4,096 inputs and 4,096 outputs. Suppose we want to change every diagonal entry and leave every other entry untouched:

<div class="math-display" markdown="0">
\[
\Delta W=I_{4096}.
\]
</div>

This update changes 4,096 connections. Its rank is also 4,096.

A coordinate representation expresses it with 4,096 trainable values. A standard LoRA parameterization with rank 16 uses 131,072 factor values and leaves 99.6% of the target's squared Frobenius norm unexplained. To represent the target exactly, the two LoRA factors require rank 4,096 and contain 33,554,432 values.

The difference comes from the structure of the update. A small number of independent connection changes can form a full-rank matrix.

**Parameter efficiency and low rank are different properties.**

SparseLeaf starts from this distinction. It freezes a pretrained network, gives a selected set of existing connections independent trainable increments, and optimizes those increments directly. The entire network supplies its computation; a small set of connections supplies its plasticity.

We call this **synaptic sparse fine-tuning**. Its central principle is simple:

> Constrain how many connections can change. Preserve the freedom for those changes to form a high-rank update.

<span id="1-why-lora-is-wrong-parameter-efficiency-is-not-low-rank"></span>

## 1. Why LoRA Is Wrong: The Cost of Factorization

LoRA's theoretical cost appears in three places: the updates it can represent, the directions it can learn at initialization, and the geometry of its optimizer. We examine these together: first the rank constraint and its task-loss consequence, then the initialization and optimization mechanisms, and finally the direct-coordinate alternative.

### 1.1 Expressivity: low rank is not sparsity

#### Two different constraints

Write the adapted weight matrix as

<div class="math-display" markdown="0">
\[
W=W_0+\Delta W,
\]
</div>

where <span class="math-inline" markdown="0">\(W_0\)</span> is the pretrained weight and <span class="math-inline" markdown="0">\(\Delta W\)</span> is the accumulated change introduced by learning.

There are two distinct ways to constrain this change:

<div class="math-display" markdown="0">
\[
\operatorname{rank}(\Delta W)\le r
\qquad\text{and}\qquad
\&#124;\Delta W\&#124;_0\le K.
\]
</div>

The first limits the number of independent matrix directions. The second limits the number of weight coordinates that change.

A dense outer product <span class="math-inline" markdown="0">\(uv^\top\)</span> can modify every entry while having rank one. A diagonal matrix with nonzero diagonal entries changes very few entries while having full rank. Sparsity describes where changes occur; rank describes how many independent directions they contain.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/01-sparsity-and-rank.png" alt="A dense rank-one outer product and a sparse full-rank diagonal update, each shown as an eight-by-eight matrix." width="2560" height="1280" loading="eager" fetchpriority="high" decoding="async">

*Figure 1. Two different structures. The outer product changes all 64 entries and has rank one. The diagonal update changes eight entries and has rank eight. These are analytic examples of update matrices.*

#### The expressive price of factorization

LoRA represents the update to an <span class="math-inline" markdown="0">\(m\times n\)</span> weight matrix through two trainable factors:

<div class="math-display" markdown="0">
\[
\Delta W=BA,
\qquad
B\in\mathbb R^{m\times r},
\quad
A\in\mathbb R^{r\times n}.
\]
</div>

It stores <span class="math-inline" markdown="0">\(r(m+n)\)</span> trainable factor values and enforces <span class="math-inline" markdown="0">\(\operatorname{rank}(\Delta W)\le r\)</span>. This is the low-rank parameterization introduced by Hu and colleagues in [LoRA](https://arxiv.org/abs/2106.09685).

For an input <span class="math-inline" markdown="0">\(x\)</span>, the output correction is

<div class="math-display" markdown="0">
\[
\Delta y=B(Ax).
\]
</div>

Every correction passes through an <span class="math-inline" markdown="0">\(r\)</span>-dimensional intermediate representation and lies in the column space of <span class="math-inline" markdown="0">\(B\)</span>. Training can rotate that space; its dimension remains at most <span class="math-inline" markdown="0">\(r\)</span>.

The factors also couple the changes made to individual connections. Changing one entry of <span class="math-inline" markdown="0">\(B\)</span> changes a row pattern determined by <span class="math-inline" markdown="0">\(A\)</span>. Changing one entry of <span class="math-inline" markdown="0">\(A\)</span> changes a column pattern determined by <span class="math-inline" markdown="0">\(B\)</span>. Learning operates through shared patterns of connection changes.

Now consider the family

<div class="math-display" markdown="0">
\[
\Delta W^*=\operatorname{diag}(a_1,\ldots,a_d),
\qquad a_i\ne0.
\]
</div>

Each of its <span class="math-inline" markdown="0">\(d\)</span> connections has an independent adjustment. The family has <span class="math-inline" markdown="0">\(d\)</span> scalar degrees of freedom, and every member has rank <span class="math-inline" markdown="0">\(d\)</span>.

Coordinate learning assigns one variable to each adjustment. Standard LoRA needs <span class="math-inline" markdown="0">\(r\ge d\)</span> to express these matrices exactly, at which point its two factors contain at least <span class="math-inline" markdown="0">\(2d^2\)</span> values.

This is the central criticism: **a low-rank parameterization turns a restriction on learning variables into a restriction on independent update directions.** For sparse, independent corrections, rank is the wrong quantity to ration.

The expressive loss has an exact form. By the [Eckart–Young low-rank approximation theorem](https://www.cambridge.org/core/journals/psychometrika/article/approximation-of-one-matrix-by-another-of-lower-rank/B29672E1EDD0FA1B7611D4DFAFC321B3), if the target update has singular values <span class="math-inline" markdown="0">\(\sigma_1\ge\sigma_2\ge\cdots\)</span>, its best rank-<span class="math-inline" markdown="0">\(r\)</span> approximation satisfies

<div class="math-display" markdown="0">
\[
\min_{\operatorname{rank}(M)\le r}
\&#124;\Delta W^*-M\&#124;_F^2
=\sum_{i&gt;r}\sigma_i^2.
\]
</div>

The discarded singular directions have a price even at the best possible solution. Zeng and Lee make this relationship explicit in their analysis of LoRA's expressive power: their deep-linear theorem characterizes the optimal approximation error through the singular spectrum of the target correction and identifies the adapter rank needed for exact representation. Rank is an expressivity budget. [The Expressive Power of Low-Rank Adaptation, §2](https://arxiv.org/html/2310.17513)

For <span class="math-inline" markdown="0">\(\Delta W^*=aI_d\)</span>, all <span class="math-inline" markdown="0">\(d\)</span> singular values equal <span class="math-inline" markdown="0">\(&#124;a&#124;\)</span>. The fraction of squared update energy left out is

<div class="math-display" markdown="0">
\[
1-\frac rd.
\]
</div>

#### Low intrinsic dimension is not low matrix rank

The literature on intrinsic dimension asks how many trainable variables are needed to adapt a pretrained model. Aghajanyan, Gupta, and Zettlemoyer report that 200 variables, mapped into the full parameter space through a random projection, achieve 90% of full-parameter fine-tuning performance on MRPC with RoBERTa. Their experiments also show that pretraining reduces the measured intrinsic dimension of adaptation. [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://aclanthology.org/2021.acl-long.568/)

This is a statement about the dimension of the learning parameterization. Matrix rank measures something else: the number of independent directions in a particular layer's update. Our diagonal family makes the separation explicit—<span class="math-inline" markdown="0">\(d\)</span> learning variables produce rank-<span class="math-inline" markdown="0">\(d\)</span> matrices. **A compact adaptation can carry a high-rank transformation.**

Independent connection adjustments do not become redundant merely because we want a small training state. SparseLeaf keeps those adjustments independent and places the budget on their count.

<span id="3-when-high-rank-sparse-learning-beats-low-rank-dense-learning"></span>

### 1.2 A rank constraint becomes a task-loss floor

We can turn the expressive difference into an exact learning result.

Consider a <span class="math-inline" markdown="0">\(d\times d\)</span> linear layer. Draw inputs with second moment

<div class="math-display" markdown="0">
\[
\mathbb E[xx^\top]=I_d,
\]
</div>

and define the target by

<div class="math-display" markdown="0">
\[
y=(W_0+aI_d)x,
\qquad a\ne0.
\]
</div>

Every input coordinate requires its own adjustment. With squared loss,

<div class="math-display" markdown="0">
\[
\begin{aligned}
\mathcal L(W)
&amp;=\frac12\mathbb E\&#124;Wx-y\&#124;_2^2\\
&amp;=\frac12\&#124;W-W_0-aI_d\&#124;_F^2.
\end{aligned}
\]
</div>

#### SparseLeaf follows the full-gradient trajectory

Select the <span class="math-inline" markdown="0">\(d\)</span> diagonal coordinates and initialize all increments at zero. Use the exact gradient of this population loss, ordinary gradient descent, and a constant step size <span class="math-inline" markdown="0">\(0&lt;\eta&lt;2\)</span>.

Each selected increment follows

<div class="math-display" markdown="0">
\[
\theta_{i,t+1}
=\theta_{i,t}-\eta(\theta_{i,t}-a).
\]
</div>

Therefore,

<div class="math-display" markdown="0">
\[
\theta_{i,t}=a\bigl[1-(1-\eta)^t\bigr],
\]
</div>

and

<div class="math-display" markdown="0">
\[
\boxed{
\mathcal L_t^{\mathrm{SparseLeaf}}
=\mathcal L_t^{\mathrm{Full}}
=\frac{da^2}{2}(1-\eta)^{2t}
\longrightarrow0.
}
\]
</div>

The full gradient is diagonal throughout this trajectory. SparseLeaf learns every coordinate that the full update changes, producing the same weights at every step.

#### Low rank leaves a positive minimum loss

A LoRA update satisfies <span class="math-inline" markdown="0">\(\operatorname{rank}(BA)\le r\)</span>. For <span class="math-inline" markdown="0">\(r&lt;d\)</span>, the best achievable loss is

<div class="math-display" markdown="0">
\[
\boxed{
\min_{A,B}\mathcal L(W_0+BA)
=\frac{d-r}{2}a^2.
}
\]
</div>

This is the global minimum over the factors. Optimization within the rank-<span class="math-inline" markdown="0">\(r\)</span> family cannot remove the error associated with the remaining <span class="math-inline" markdown="0">\(d-r\)</span> directions.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/03-task-loss-separation.png" alt="Exact SparseLeaf and full-gradient loss trajectories, together with the global minimum loss imposed by a rank-eight update in a 64-dimensional diagonal regression task." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 2. Analytic regression example with <span class="math-inline" markdown="0">\(d=64\)</span>, <span class="math-inline" markdown="0">\(r=8\)</span>, <span class="math-inline" markdown="0">\(a=1\)</span>, and <span class="math-inline" markdown="0">\(\eta=0.1\)</span>. SparseLeaf and full-gradient descent have the same trajectory. The orange line is the best loss achievable by any rank-eight update.*

The comparison is unusually direct. SparseLeaf learns <span class="math-inline" markdown="0">\(d\)</span> values and reaches zero loss. Standard LoRA learns <span class="math-inline" markdown="0">\(2dr\)</span> factor values and retains a positive loss floor.

The target is simple in coordinates and rich in independent directions. A sparse parameterization captures both facts at once.

**High-rank sparse learning wins here because it allocates variables to the changes the task actually requires.**

#### The empirical counterpart

Sparse high-rank adaptation also wins direct comparisons in language models. In Bhardwaj and colleagues' SHiRA experiments, LLaMA-7B reaches 77.4% average accuracy across eight commonsense-reasoning benchmarks with SHiRA-SNIP, compared with 74.7% for LoRA. SHiRA trains 1.0% of the original parameters; LoRA uses 0.83% in trainable factors. The magnitude-selected SHiRA variant reaches 77.0%. [Sparse High Rank Adapters, Table 2](https://arxiv.org/html/2406.13175)

Liu and colleagues report a matched-parameter comparison on Gemma2-2B, trained on MetaMathQA and evaluated on five-shot GSM8K. Their static, gradient-selected sparse adaptation scores 50.27% versus LoRA's 39.20% with flexible answer extraction, and 37.15% versus 28.81% with strict matching. The trainable budget is held equal while its parameterization changes. [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models, Table 3](https://aclanthology.org/2025.acl-long.1541/)

The analytic task explains an expressive advantage of sparse, independent corrections. These published comparisons show that coordinate-sparse adaptation can turn its learning budget into better task performance than low-rank factorization.

### 1.3 Initialization: which directions can learn first?

#### Zero output can hide an inactive learning path

Factorization imposes a second cost before learning has made its first update. The standard initialization sets <span class="math-inline" markdown="0">\(B_0=0\)</span> and draws <span class="math-inline" markdown="0">\(A_0\)</span> randomly, making <span class="math-inline" markdown="0">\(B_0A_0=0\)</span>. This is the zero-product initialization specified in the original LoRA paper. [Hu et al., §4.1](https://arxiv.org/abs/2106.09685) It preserves the pretrained model's initial function. It does not preserve its learning geometry.

Take unit adapter scaling and write <span class="math-inline" markdown="0">\(G=\nabla_W\mathcal L\)</span>. The factor gradients are

<div class="math-display" markdown="0">
\[
\nabla_B\mathcal L=GA^\top,\qquad
\nabla_A\mathcal L=B^\top G.
\]
</div>

At initialization,

<div class="math-display" markdown="0">
\[
\nabla_B\mathcal L=GA_0^\top,\qquad
\nabla_A\mathcal L=0.
\]
</div>

The entire <span class="math-inline" markdown="0">\(A\)</span> block receives no task gradient on the first backward pass. Learning initially changes how the random features are combined through <span class="math-inline" markdown="0">\(B\)</span>; changing the features through <span class="math-inline" markdown="0">\(A\)</span> depends on <span class="math-inline" markdown="0">\(B\)</span> first becoming nonzero.

Setting both factors to zero makes both gradients zero. Gradient descent and Adam with zero initial moments then remain at that point. A zero *product* and a trainable zero *increment* are fundamentally different constructions.

The differential makes the inactive path explicit:

<div class="math-display" markdown="0">
\[
\mathrm d(BA)=(\mathrm dB)A+B(\mathrm dA)
\quad\Longrightarrow\quad
\left.\mathrm d(BA)\right&#124;_{B_0=0}
=(\mathrm dB)A_0.
\]
</div>

For a full-row-rank <span class="math-inline" markdown="0">\(A_0\)</span>, the initial parameter-to-weight Jacobian has rank <span class="math-inline" markdown="0">\(mr\)</span>. LoRA stores <span class="math-inline" markdown="0">\(r(m+n)\)</span> factor values, but all <span class="math-inline" markdown="0">\(rn\)</span> coordinates of <span class="math-inline" markdown="0">\(A\)</span> are invisible to this initial differential. The parameter count overstates the number of independent directions available at the starting point.

#### The random basis decides what can move first

The initial tangent space—the weight changes available to first order—is

<div class="math-display" markdown="0">
\[
\mathcal T_0
=\{HA_0:H\in\mathbb R^{m\times r}\}.
\]
</div>

Every row of an initial correction lies in the row space of <span class="math-inline" markdown="0">\(A_0\)</span>. For an input direction <span class="math-inline" markdown="0">\(v\)</span> with <span class="math-inline" markdown="0">\(A_0v=0\)</span>, every correction in <span class="math-inline" markdown="0">\(\mathcal T_0\)</span> satisfies <span class="math-inline" markdown="0">\((HA_0)v=0\)</span>. At rank <span class="math-inline" markdown="0">\(r&lt;n\)</span>, there are at least <span class="math-inline" markdown="0">\(n-r\)</span> such independent input directions.

This is an initialization bottleneck as well as a rank bottleneck: the first corrections must pass through a randomly chosen set of input directions. The task cannot immediately use arbitrary directions within the eventual rank-<span class="math-inline" markdown="0">\(r\)</span> model family. Learning the basis itself starts through the other factor's learning.

Even exchanging the zero and random factors changes the dynamics. Hayou, Ghosh, and Yu derive different stable learning-rate scalings and observe different fine-tuning performance for these two initializations, despite identical initial model outputs. **Starting from the same function does not mean starting with the same ability to learn.** [The Impact of Initialization on LoRA Finetuning Dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html)

#### Initialization determines the first learning geometry

The initial basis also determines the geometry of the first update. Take unit adapter scaling, no adapter dropout, and ordinary gradient descent. At the base model, let <span class="math-inline" markdown="0">\(G=\nabla_W\mathcal L(W_0)\)</span>, draw <span class="math-inline" markdown="0">\(A_0\)</span> randomly, and set <span class="math-inline" markdown="0">\(B_0=0\)</span>. Since the first task gradient of <span class="math-inline" markdown="0">\(A\)</span> is zero, <span class="math-inline" markdown="0">\(A_1=A_0\)</span>, while <span class="math-inline" markdown="0">\(B_1=-\eta GA_0^\top\)</span>. The first effective weight update is exactly

<div class="math-display" markdown="0">
\[
\Delta W_1=-\eta GA_0^\top A_0.
\]
</div>

The gradient passes through the random Gram matrix <span class="math-inline" markdown="0">\(A_0^\top A_0\)</span>. Its scale and directional weighting come from the initialization. Replacing <span class="math-inline" markdown="0">\(A_0\)</span> with <span class="math-inline" markdown="0">\(cA_0\)</span> leaves the initial model unchanged but multiplies this SGD update by <span class="math-inline" markdown="0">\(c^2\)</span> at the same learning rate. Initialization therefore changes the effective optimizer while keeping the starting predictions identical.

Flora develops this connection into a random-projection interpretation of LoRA's SGD dynamics. It then refreshes the projections to obtain high-rank accumulated updates while keeping optimizer states compact. The distinction is fundamental: compressing the state used to learn and restricting the rank of the accumulated change are different design decisions. [Flora, §§2–3](https://proceedings.mlr.press/v235/hao24a.html)

To isolate random geometry from mean scale, use independent <span class="math-inline" markdown="0">\(A_{0,ij}\sim\mathcal N(0,1/r)\)</span> and fixed <span class="math-inline" markdown="0">\(G\)</span>. This normalization gives <span class="math-inline" markdown="0">\(\mathbb E[A_0^\top A_0]=I_n\)</span>, so the mean first update is <span class="math-inline" markdown="0">\(-\eta G\)</span>. Its mean squared deviation from that update is

<div class="math-display" markdown="0">
\[
\mathbb E\&#124;\Delta W_1+\eta G\&#124;_F^2
=\eta^2\frac{n+1}{r}\&#124;G\&#124;_F^2.
\]
</div>

For <span class="math-inline" markdown="0">\(n=4096\)</span> and <span class="math-inline" markdown="0">\(r=8\)</span>, the relative mean squared deviation is <span class="math-inline" markdown="0">\(4097/8=512.125\)</span>. Matching the full-gradient step *in expectation* says little about the step produced by one initialized adapter. The expandable derivation below gives the exact calculation.

#### Better scaling cannot restore missing directions

Remove the random singular-value scaling and examine the initial subspace itself. For full-row-rank <span class="math-inline" markdown="0">\(A_0\)</span>, its orthogonal projector is

<div class="math-display" markdown="0">
\[
\Pi_0=A_0^\top(A_0A_0^\top)^{-1}A_0.
\]
</div>

The closest approximation to <span class="math-inline" markdown="0">\(G\)</span> inside <span class="math-inline" markdown="0">\(\mathcal T_0\)</span> is <span class="math-inline" markdown="0">\(G\Pi_0\)</span>. An isotropic Gaussian initialization chooses a uniformly oriented <span class="math-inline" markdown="0">\(r\)</span>-dimensional row space, giving

<div class="math-display" markdown="0">
\[
\mathbb E\&#124;G\Pi_0\&#124;_F^2=\frac rn\&#124;G\&#124;_F^2,
\qquad
\mathbb E\&#124;G-G\Pi_0\&#124;_F^2
=\left(1-\frac rn\right)\&#124;G\&#124;_F^2.
\]
</div>

At width 4096 and rank 8, the initial tangent space captures 0.1953% of a fixed task gradient's squared energy on average. This is the best projection into that initial space, with its conditioning already removed. Rescaling a step cannot recover the missing directions; they require learning a different basis.

<details id="appendix-a-the-first-step-geometry-of-gaussian-lora-initialization" markdown="1">
<summary>Derivation: Gaussian first-step error and projected gradient energy</summary>

Take a deterministic forward pass without adapter dropout, unit adapter scaling, ordinary gradient descent, <span class="math-inline" markdown="0">\(B_0=0\)</span>, and independent entries <span class="math-inline" markdown="0">\(A_{0,ij}\sim\mathcal N(0,1/r)\)</span>. The base-model gradient <span class="math-inline" markdown="0">\(G\)</span> is fixed and independent of <span class="math-inline" markdown="0">\(A_0\)</span>.

Write

<div class="math-display" markdown="0">
\[
Q=A_0^\top A_0
=\frac1r\sum_{\ell=1}^{r}z_\ell z_\ell^\top,
\qquad
z_\ell\sim\mathcal N(0,I_n).
\]
</div>

Then <span class="math-inline" markdown="0">\(\mathbb E Q=I_n\)</span>. For one Gaussian vector,

<div class="math-display" markdown="0">
\[
\mathbb E(zz^\top)^2=(n+2)I_n,
\]
</div>

because each diagonal entry has expectation

<div class="math-display" markdown="0">
\[
\mathbb E\left[z_i^2\sum_j z_j^2\right]
=3+(n-1)=n+2,
\]
</div>

and off-diagonal expectations vanish. Independence gives

<div class="math-display" markdown="0">
\[
\mathbb E Q^2
=\frac{r(n+2)+r(r-1)}{r^2}I_n
=\left(1+\frac{n+1}{r}\right)I_n.
\]
</div>

Consequently,

<div class="math-display" markdown="0">
\[
\mathbb E(Q-I_n)^2=\frac{n+1}{r}I_n.
\]
</div>

The effective first update is <span class="math-inline" markdown="0">\(\Delta W_1=-\eta GQ\)</span>, so

<div class="math-display" markdown="0">
\[
\begin{aligned}
\mathbb E\&#124;\Delta W_1+\eta G\&#124;_F^2
&amp;=\eta^2\operatorname{tr}
\left(G\,\mathbb E[(Q-I_n)^2]G^\top\right)\\
&amp;=\eta^2\frac{n+1}{r}\&#124;G\&#124;_F^2.
\end{aligned}
\]
</div>

Averaging <span class="math-inline" markdown="0">\(M\)</span> independent initialized updates divides this mean squared deviation by <span class="math-inline" markdown="0">\(M\)</span>. The factorization thus has both an exact mean update and a computable distribution around it.

#### The initial-subspace projection

For <span class="math-inline" markdown="0">\(1\le r\le n\)</span>, a Gaussian <span class="math-inline" markdown="0">\(A_0\)</span> has full row rank almost surely. Its row-space projector <span class="math-inline" markdown="0">\(\Pi_0\)</span> satisfies <span class="math-inline" markdown="0">\(\Pi_0^\top=\Pi_0\)</span>, <span class="math-inline" markdown="0">\(\Pi_0^2=\Pi_0\)</span>, and <span class="math-inline" markdown="0">\(\operatorname{tr}\Pi_0=r\)</span>. Rotational invariance implies <span class="math-inline" markdown="0">\(\mathbb E\Pi_0=cI_n\)</span>; taking traces gives <span class="math-inline" markdown="0">\(c=r/n\)</span>.

Orthogonal projection onto <span class="math-inline" markdown="0">\(\mathcal T_0=\{HA_0\}\)</span> acts on each row of <span class="math-inline" markdown="0">\(G\)</span>, so

<div class="math-display" markdown="0">
\[
\arg\min_{M\in\mathcal T_0}\&#124;G-M\&#124;_F^2=G\Pi_0.
\]
</div>

Idempotence then yields

<div class="math-display" markdown="0">
\[
\begin{aligned}
\mathbb E\&#124;G\Pi_0\&#124;_F^2
&amp;=\operatorname{tr}(G\,\mathbb E\Pi_0\,G^\top)
=\frac rn\&#124;G\&#124;_F^2,\\
\mathbb E\&#124;G-G\Pi_0\&#124;_F^2
&amp;=\left(1-\frac rn\right)\&#124;G\&#124;_F^2.
\end{aligned}
\]
</div>

This isolates the missing-direction cost from the random scaling of the unnormalized Gram matrix.

</details>

### 1.4 Optimization: factors change the meaning of a step

#### Rank and update scale are entangled

The adapter's external multiplier introduces another scale coupling. Write the adapter as <span class="math-inline" markdown="0">\(\gamma_r BA\)</span>, keep <span class="math-inline" markdown="0">\(B_0=0\)</span>, and now use independent, zero-mean entries of <span class="math-inline" markdown="0">\(A_0\)</span> with variance <span class="math-inline" markdown="0">\(\sigma_A^2\)</span> independent of <span class="math-inline" markdown="0">\(r\)</span>. The same first-step SGD calculation gives

<div class="math-display" markdown="0">
\[
\mathbb E[\Delta W_1]
=-\eta\gamma_r^2 r\sigma_A^2G.
\]
</div>

With fixed <span class="math-inline" markdown="0">\(\alpha\)</span>, the usual <span class="math-inline" markdown="0">\(\gamma_r=\alpha/r\)</span> makes this mean step shrink as <span class="math-inline" markdown="0">\(1/r\)</span>; <span class="math-inline" markdown="0">\(\gamma_r=\alpha/\sqrt r\)</span> keeps its scale constant. The rsLoRA paper derives the square-root rule through a rank-stability analysis and demonstrates improved use of larger ranks. Rank is entangled with optimization scale before its additional capacity can be used. [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA, §3 and Appendix A](https://arxiv.org/html/2312.03732)

#### Learning the basis couples the two factor learning rates

The two factors also control that basis learning through one another. With SGD learning rates <span class="math-inline" markdown="0">\(\eta_A,\eta_B\)</span> and <span class="math-inline" markdown="0">\(G_t=\nabla_W\mathcal L(W_t)\)</span>, the second update gives

<div class="math-display" markdown="0">
\[
A_2-A_1=\eta_A\eta_B A_0G_0^\top G_1.
\]
</div>

Its onset is coupled to both learning rates. LoRA+ analyzes this imbalance in the large-width limit and uses different factor learning rates to improve feature learning. The factors require a relative learning-rate choice in addition to the choice of starting basis. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://proceedings.mlr.press/v235/hayou24a.html)

#### Adam normalizes factor gradients inside the same initial subspace

The first-step geometry remains explicit with Adam. Start both factors' optimizer moments at zero and define <span class="math-inline" markdown="0">\(\Gamma_0=GA_0^\top\)</span>. Bias correction gives <span class="math-inline" markdown="0">\(\widehat m_{B,1}=\Gamma_0\)</span> and <span class="math-inline" markdown="0">\(\widehat v_{B,1}=\Gamma_0^{\odot2}\)</span>. The resulting unit-scaled adapter is

<div class="math-display" markdown="0">
\[
\Delta W_1^{\mathrm{LoRA\text{-}Adam}}
=-\eta_B
\left(\frac{\Gamma_0}{&#124;\Gamma_0&#124;+\epsilon}\right)A_0.
\]
</div>

Adam changes the coefficients multiplying <span class="math-inline" markdown="0">\(A_0\)</span>. Every row of this first update still belongs to its initial row space, and <span class="math-inline" markdown="0">\(A\)</span> still receives zero task gradient at the starting point. Factorwise normalization does not create directions absent from that space.

There is a second problem with factorwise normalization. The same product has equivalent representations <span class="math-inline" markdown="0">\(BA=(BR)(R^{-1}A)\)</span> for any invertible <span class="math-inline" markdown="0">\(R\)</span>. Yen and colleagues show that gradient descent and Adam generally produce different effective weight updates from equivalent factorizations. Their LoRA-RITE optimizer introduces matrix preconditioning to restore transformation invariance. **Normalizing factor gradients is not the same as normalizing changes to the model.** [LoRA Done RITE, §§2–3](https://arxiv.org/html/2410.20625)

### 1.5 What initialization and optimizer research is repairing

LoRA-Pro states the optimization problem directly in weight space. It adjusts factor gradients to minimize the difference between their induced weight differential and the full-fine-tuning gradient. The factorized parameterization introduces a gradient-approximation problem inside the learning procedure itself. [LoRA-Pro, §§2.2–2.3](https://arxiv.org/html/2407.18242)

LoRA-GA explicitly targets the mismatch between LoRA's initial weight-update direction and the full-fine-tuning gradient. It initializes the factors from singular vectors of a sampled task gradient and offsets the frozen weight to preserve the initial model. [LoRA-GA, §§3.2–3.4](https://arxiv.org/html/2407.05000)

PiSSA instead initializes the factors from principal components of the pretrained weight and freezes its residual. The initial function is preserved, but the basis now comes from the pretrained matrix rather than a random draw. [PiSSA, §3](https://arxiv.org/html/2404.02948)

EVA uses a third source of directions: the pretrained model's input activations on task data. It initializes <span class="math-inline" markdown="0">\(A\)</span> from their leading right singular vectors and keeps <span class="math-inline" markdown="0">\(B_0=0\)</span>. Its <span class="math-inline" markdown="0">\(\rho=1\)</span> configuration retains uniform ranks, giving a direct comparison of activation-based and random initialization at the same parameter count. [Explained Variance Adaptation, §§3.2–3.5](https://arxiv.org/html/2410.07170)

Published experiments make the consequences concrete. Each row below is a comparison within the cited study; the reported results are task accuracies.

| Factorization choice tested | Model and training setup | Reported result |
|---|---|---|
| Which factor starts at zero | RoBERTa-large on MNLI; <span class="math-inline" markdown="0">\(r=8\)</span>, FP16, three seeds; learning rate searched for each initialization | <span class="math-inline" markdown="0">\(A_0=0\)</span>: 89.47%; <span class="math-inline" markdown="0">\(B_0=0\)</span>: 90.69%. [Hayou et al., §4.1 and Figure 4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html) |
| Gradient-aligned initialization and stable scaling | Llama-2-7B, MetaMathQA training, GSM8K evaluation; <span class="math-inline" markdown="0">\(r=8\)</span>, three seeds | LoRA: 42.08% → LoRA-GA: 53.60%. [LoRA-GA, Table 2](https://arxiv.org/html/2407.05000) |
| Activation-based initialization at fixed rank and parameter count | Llama-2-7B, MetaMathQA training, GSM8K evaluation; <span class="math-inline" markdown="0">\(r=16\)</span>, 40.6M trainable parameters each, three seeds | LoRA: 59.7% → EVA (<span class="math-inline" markdown="0">\(\rho=1\)</span>): 61.9%. [EVA, Table 11](https://arxiv.org/html/2410.07170) |
| Transformation-invariant factor optimization | Gemma-7B on GSM8K; <span class="math-inline" markdown="0">\(r=16\)</span>; learning rate searched for each optimizer | Adam: 48.37% → LoRA-RITE: 55.50%. [LoRA-RITE, §5 and Table 2](https://arxiv.org/html/2410.20625) |

The difference is visible in learned weights as well. Shuttleworth and colleagues find that LoRA and full fine-tuning produce distinct spectral structures even at similar downstream performance. LoRA introduces prominent singular vectors that differ sharply from those of the pretrained model, which the authors call *intruder dimensions*. A parameterization shapes both the path of learning and the structure of its solution. [LoRA vs Full Fine-tuning: An Illusion of Equivalence, §3](https://arxiv.org/html/2410.21228v3)

### 1.6 The alternative: independent connection increments

SparseLeaf gives each selected connection its own increment:

<div class="math-display" markdown="0">
\[
\Delta W(\theta)=\sum_k\theta_k e_{i_k}e_{j_k}^\top,
\qquad
\theta_0=0.
\]
</div>

Its derivative is already present at zero:

<div class="math-display" markdown="0">
\[
\frac{\partial W}{\partial\theta_k}
=e_{i_k}e_{j_k}^\top,\qquad
\left.\frac{\partial\mathcal L}{\partial\theta_k}\right&#124;_{\theta=0}
=G_{i_k,j_k}.
\]
</div>

All selected connection directions are available from the first backward pass. Zero increments preserve the pretrained function without silencing a parameter block or introducing a random learned basis between the task gradient and its trainable variables.

The support determines where learning is allowed. The native gradient determines how each permitted connection should change. SparseLeaf starts with both parts of that interface intact.

For a flattened gradient <span class="math-inline" markdown="0">\(g\)</span>, let <span class="math-inline" markdown="0">\(P_S\)</span> extract the selected coordinates. An ordinary coordinate-gradient step is <span class="math-inline" markdown="0">\(-\eta P_S^\top P_Sg\)</span>. The extraction preserves each selected component instead of mixing it through a learned factor.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/04-native-coordinate-learning.png" alt="Two first-step learning paths: LoRA transforms the weight gradient through its initialized factors, while SparseLeaf extracts the selected native coordinates." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 3. Both parameterizations start at the base model. With unit adapter scaling and ordinary gradient descent, LoRA's first effective update is <span class="math-inline" markdown="0">\(-\eta GA_0^\top A_0\)</span>; SparseLeaf's is <span class="math-inline" markdown="0">\(-\eta P_S^\top P_Sg\)</span> in flattened coordinates.*

With Adam, let <span class="math-inline" markdown="0">\(g_S=P_Sg\)</span> at the same pretrained model. Its first Adam update is

<div class="math-display" markdown="0">
\[
\theta_1=-\eta\frac{g_S}{&#124;g_S&#124;+\epsilon},
\qquad
\delta w_1=P_S^\top\theta_1.
\]
</div>

Every selected connection is normalized using its own task gradient. The embedding has <span class="math-inline" markdown="0">\(K\)</span> orthonormal coordinate directions even at <span class="math-inline" markdown="0">\(\theta=0\)</span>. There is no factor whose zero value blocks another factor's learning, and no random Gram matrix to insert between the selected gradient and its update. For a fixed support and coordinate ordering, each sparse increment has one parameter vector; there is no equivalent-factor scaling or rotation for the optimizer to reconcile.

Together with LoRA+, these studies identify the optimization work introduced by factorization: choose a basis, calibrate its scale, balance its factors, and correct the geometry of their updates. SparseLeaf places the design decision directly on the support. Once the connections are selected, zero increments and zero Adam moments are enough to start learning in their native coordinates. **Preserve the pretrained computation, and spend the learning budget on independently changing its connections.**

## 2. SparseLeaf: Give Existing Connections Plasticity

In a neural layer,

<div class="math-display" markdown="0">
\[
z_i=\sum_j W_{ij}x_j,
\qquad h_i=\phi(z_i),
\]
</div>

the scalar <span class="math-inline" markdown="0">\(W_{ij}\)</span> is the connection from input activation <span class="math-inline" markdown="0">\(x_j\)</span> to output unit <span class="math-inline" markdown="0">\(i\)</span>. Selecting that coordinate for training gives that connection the ability to change its strength.

The synaptic perspective has a concrete biological motivation. In hippocampal neurons, Matsuzaki and colleagues induced potentiation at individual dendritic spines: structural enlargement and increased AMPA-receptor currents were localized to the stimulated spine, with neighboring spines unaffected. The individual connection can be a unit of plasticity. SparseLeaf takes that connection-level viewpoint as its organizing principle for fine-tuning. [Structural basis of long-term potentiation in single dendritic spines](https://pmc.ncbi.nlm.nih.gov/articles/PMC4158816/)

For a frozen matrix <span class="math-inline" markdown="0">\(W_0\in\mathbb R^{m\times n}\)</span>, choose <span class="math-inline" markdown="0">\(K\)</span> distinct coordinates:

<div class="math-display" markdown="0">
\[
S=\{(i_k,j_k)\}_{k=1}^K.
\]
</div>

Associate an independent increment <span class="math-inline" markdown="0">\(\theta_k\)</span> with each selected connection:

<div class="math-display" markdown="0">
\[
\boxed{
\Delta W(\theta)=\sum_{k=1}^K\theta_k e_{i_k}e_{j_k}^{\top},
\qquad W=W_0+\Delta W(\theta).
}
\]
</div>

The trainable vector is <span class="math-inline" markdown="0">\(\theta\in\mathbb R^K\)</span>. We initialize it at zero, so learning starts from the pretrained model.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/02-synaptic-plasticity.png" alt="Three selected neural connections mapped to three weight coordinates and three independent trainable increments." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 4. One set of connections, three views. Gray connections retain their pretrained weights and participate in computation. Colored connections use their pretrained weights plus independent increments. The same colors identify the connections, matrix entries, and trainable values.*

This separates two roles that are often bundled together. The dense network provides the representational machinery built during pretraining. The selected connections provide the degrees of freedom used to adapt that machinery.

**Dense capacity. Sparse plasticity.**

### Sparsity as an optimization constraint

The underlying optimization problem is

<div class="math-display" markdown="0">
\[
\min_{\Delta W}\mathcal L(W_0+\Delta W)
\quad\text{subject to}\quad
\&#124;\Delta W\&#124;_0\le K.
\]
</div>

There are two decisions inside this problem: where plasticity is allocated, and what values the plastic connections learn. SparseLeaf makes these decisions explicit. It chooses a support <span class="math-inline" markdown="0">\(S\)</span>, then solves

<div class="math-display" markdown="0">
\[
\min_{\theta\in\mathbb R^K}
\mathcal L\!\left(W_0+
\sum_{k=1}^K\theta_k e_{i_k}e_{j_k}^{\top}\right).
\]
</div>

For a fixed support, this is learning in a coordinate subspace. Each trainable direction is an original weight coordinate, and each variable controls exactly one connection.

The largest rank attainable by an <span class="math-inline" markdown="0">\(m\times n\)</span> matrix with at most <span class="math-inline" markdown="0">\(K\)</span> nonzero entries is

<div class="math-display" markdown="0">
\[
\max_{\&#124;M\&#124;_0\le K}\operatorname{rank}(M)
=\min(K,m,n).
\]
</div>

Placing nonzero entries on distinct rows and columns attains this maximum. In a square layer, a diagonal support reaches full rank with only one trainable connection per row and column.

The support remains fixed during training, so both the accumulated change <span class="math-inline" markdown="0">\(W_t-W_0\)</span> and every step <span class="math-inline" markdown="0">\(W_{t+1}-W_t\)</span> stay within the same set of <span class="math-inline" markdown="0">\(K\)</span> connections. The sparsity budget organizes the entire learning trajectory.

### Sparse changes are a learning primitive

Diff Pruning formulates adaptation as a sparse additive difference from frozen pretrained parameters, using a differentiable approximation to an <span class="math-inline" markdown="0">\(L_0\)</span> penalty to learn the support. Its structured variant modifies 0.5% of BERT-Large's parameters and matches its full-fine-tuning baseline's reported GLUE average of 80.6, excluding QNLI from that average. The full network remains available while the task-specific change is sparse. [Guo, Rush, and Kim, Table 1](https://aclanthology.org/2021.acl-long.378/)

SpaRTA takes an even simpler route: randomly select a small set of original parameters and train their increments while freezing the rest. Its Gemma and Mistral experiments find competitive performance with LoRA at similar trainable-parameter counts. This provides a direct empirical example of parameter-efficient adaptation built from coordinate sparsity. [Rios et al., Sparsity May Be All You Need](https://aclanthology.org/2025.findings-emnlp.1013/)

These methods establish the usefulness of sparse changes. SparseLeaf organizes that idea around independent plastic connections, a fixed support, and an optimizer whose entire trainable state lives on that support.

<span id="4-sparseleaf-adam-optimize-the-connections-directly"></span>

## 3. SparseLeaf Adam: Optimize the Connections Directly

Expressive capacity determines which changes are available. Parameterization also determines how learning signals reach those changes.

Flatten the weights into <span class="math-inline" markdown="0">\(w\in\mathbb R^N\)</span>. Let <span class="math-inline" markdown="0">\(P_S\in\mathbb R^{K\times N}\)</span> extract the selected coordinates. SparseLeaf writes

<div class="math-display" markdown="0">
\[
w=w_0+P_S^\top\theta.
\]
</div>

The chain rule gives

<div class="math-display" markdown="0">
\[
\boxed{
\nabla_\theta\mathcal L
=P_S\nabla_w\mathcal L.
}
\]
</div>

Each increment receives the gradient of its own weight coordinate, evaluated at the current adapted model.

### The optimizer

Initialize <span class="math-inline" markdown="0">\(\theta_0=m_0=v_0=0\)</span>. At training step <span class="math-inline" markdown="0">\(t\ge1\)</span>, compute

<div class="math-display" markdown="0">
\[
g_t=P_S\nabla_w\mathcal L_t
\left(w_0+P_S^\top\theta_{t-1}\right).
\]
</div>

Maintain the first and second moments in the selected coordinate space:

<div class="math-display" markdown="0">
\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]
</div>

<div class="math-display" markdown="0">
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.
\]
</div>

Apply bias correction,

<div class="math-display" markdown="0">
\[
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
\]
</div>

and update

<div class="math-display" markdown="0">
\[
\boxed{
\theta_t=
\theta_{t-1}
-\eta_t\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\epsilon},
\qquad
w_t=w_0+P_S^\top\theta_t.
}
\]
</div>

Products, powers, square roots, and division in the optimizer are elementwise. These are [Adam's moment updates](https://arxiv.org/abs/1412.6980), applied directly to the <span class="math-inline" markdown="0">\(K\)</span> plastic connections.

There is a useful optimization interpretation. Define

<div class="math-display" markdown="0">
\[
D_t=\operatorname{diag}(\sqrt{\widehat v_t}+\epsilon).
\]
</div>

Then the increment <span class="math-inline" markdown="0">\(u_t=\theta_t-\theta_{t-1}\)</span> solves

<div class="math-display" markdown="0">
\[
u_t=\arg\min_{u\in\mathbb R^K}
\left\{
\widehat m_t^\top u+
\frac{1}{2\eta_t}u^\top D_tu
\right\}.
\]
</div>

The support specifies which connections may move. The adaptive metric specifies the scale of movement along each permitted direction. Sparsity is built into the feasible space of the optimizer.

This quadratic-step viewpoint connects SparseLeaf Adam to adaptive proximal optimization. Duchi, Hazan, and Singer developed methods in which gradient history changes the geometry used to measure a step. Here, that adaptive geometry is defined directly on the selected connections: the support chooses the axes, and each connection's second-moment history sets its metric weight. [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/v12/duchi11a.html)

### Native coordinates preserve native geometry

Because the selected coordinates are distinct,

<div class="math-display" markdown="0">
\[
P_SP_S^\top=I_K,
\qquad
\&#124;P_S^\top u\&#124;_2=\&#124;u\&#124;_2.
\]
</div>

The embedding preserves lengths and inner products. A scalar increment has the same meaning in the learning vector and in the weight matrix.

For a clean comparison, consider one ordinary gradient-descent step at the same current weights, with gradient <span class="math-inline" markdown="0">\(g\)</span>. Full-parameter gradient descent takes <span class="math-inline" markdown="0">\(-\eta g\)</span>; SparseLeaf takes <span class="math-inline" markdown="0">\(-\eta P_S^\top P_Sg\)</span>. Their squared difference is exactly

<div class="math-display" markdown="0">
\[
\&#124;\delta w_{\mathrm{SparseLeaf}}
-\delta w_{\mathrm{Full}}\&#124;_2^2
=\eta^2\sum_{i\notin S}g_i^2.
\]
</div>

Every selected coordinate receives its original gradient. The discrepancy is the gradient energy outside the selected support. When the full-gradient trajectory stays within <span class="math-inline" markdown="0">\(S\)</span>, identical initialization and training steps give identical trajectories by induction. The diagonal task above makes this identity explicit.

### Backpropagation at connection granularity

For a token batch <span class="math-inline" markdown="0">\(X\in\mathbb R^{T\times n}\)</span> and upstream gradient <span class="math-inline" markdown="0">\(D\in\mathbb R^{T\times m}\)</span>, the selected parameter gradient is

<div class="math-display" markdown="0">
\[
\frac{\partial\mathcal L}{\partial\theta_k}
=\sum_{t=1}^T D_{t,i_k}X_{t,j_k}.
\]
</div>

The implementation can compute these <span class="math-inline" markdown="0">\(K\)</span> inner products directly. Error signals continue through the full effective layer:

<div class="math-display" markdown="0">
\[
\frac{\partial\mathcal L}{\partial X}
=DW_0+\sum_k\theta_kD_{:,i_k}e_{j_k}^\top.
\]
</div>

The resulting learning rule is local in its parameterization and network-wide in its computation. Each plastic connection has its own value, gradient, and adaptive history; the surrounding network continues to participate in the forward and backward passes.

<span id="appendix-b-a-sparse-mask-can-produce-a-high-rank-step"></span>

### A sparse mask can produce a high-rank step

For one example in a linear layer, the weight gradient is an outer product:

<div class="math-display" markdown="0">
\[
G=\delta x^\top.
\]
</div>

With a binary coordinate mask <span class="math-inline" markdown="0">\(M\)</span>, a selected-coordinate gradient-descent step is

<div class="math-display" markdown="0">
\[
U=-\eta M\odot(\delta x^\top)
=-\eta\operatorname{diag}(\delta)\,
M\,\operatorname{diag}(x).
\]
</div>

Take nonzero entries in <span class="math-inline" markdown="0">\(\delta\)</span> and <span class="math-inline" markdown="0">\(x\)</span>, and nonzero <span class="math-inline" markdown="0">\(\eta\)</span>. The two diagonal matrices are invertible, giving

<div class="math-display" markdown="0">
\[
\operatorname{rank}(U)=\operatorname{rank}(M).
\]
</div>

For a square layer with <span class="math-inline" markdown="0">\(M=I_d\)</span>, the step changes <span class="math-inline" markdown="0">\(d\)</span> coordinates and has rank <span class="math-inline" markdown="0">\(d\)</span>. The coordinate mask has converted a rank-one gradient into a sparse full-rank update. Sparsity and rank describe different structures even at the level of an individual learning step.

<span id="5-which-connections-should-be-plastic"></span>

## 4. Which Connections Should Be Plastic?

Once sparsity becomes an optimization constraint, support selection becomes a question about learning: which connections can jointly produce the changes a task requires?

### Pretrained computation provides the substrate

A pretrained model already contains a large, organized system of features and transformations. Fine-tuning changes how that system responds to a task. An adjustment to an early connection can influence many downstream activations; adjustments across different layers can work together through the existing computation.

The learning signal on a connection has a precise form. For one input, write <span class="math-inline" markdown="0">\(\delta_i=\partial\mathcal L/\partial z_i\)</span>. Then

<div class="math-display" markdown="0">
\[
\frac{\partial\mathcal L}{\partial W_{ij}}
=\delta_i x_j.
\]
</div>

Across the task distribution, the persistent signal is

<div class="math-display" markdown="0">
\[
\mathbb E[\delta_i x_j].
\]
</div>

Input activity and output error jointly determine which connections receive consistent pressure to change. Adam's first moment accumulates direction over time, while its second moment tracks gradient magnitude. The learning history of each selected connection becomes part of its adaptive update rule.

### Reward selects useful changes

For on-policy learning with a parameter-independent reward, let

<div class="math-display" markdown="0">
\[
s_i=\frac{\partial\log\pi_W(y\mid x)}{\partial W_i}.
\]
</div>

The [REINFORCE score-function estimator](https://link.springer.com/article/10.1007/BF00992696) connects expected reward to connection-level updates. The score-function identity gives <span class="math-inline" markdown="0">\(\mathbb E[s_i\mid x]=0\)</span>. With the conditional mean reward as a baseline, the expected policy gradient can be written as

<div class="math-display" markdown="0">
\[
\frac{\partial J}{\partial W_i}
=\mathbb E_x\left[
\operatorname{Cov}_{y\sim\pi_W}(R,s_i\mid x)
\right].
\]
</div>

A connection receives a sustained reward-learning signal when its influence on action probabilities correlates with reward. Positive and negative contributions can cancel on other connections. This connects plasticity to the task's preferred changes in behavior.

Sparse, high-rank updates are also visible in empirical analyses of RL-trained language models. Mukherjee and colleagues examined ten models and seven RL algorithms. In their BF16 checkpoint analysis, using a <span class="math-inline" markdown="0">\(10^{-5}\)</span> change tolerance, 68.5%–96.0% of parameters remained unchanged after RL. The four model–algorithm combinations in their rank table had mean update ranks of 99.2%–99.8% of the maximum. They also retrained identified subnetworks in DPO and PRIME experiments and recovered or improved the reported task scores. These observations put sparse plasticity and high-rank updates in the same empirical picture. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models, §§3–4](https://arxiv.org/html/2505.11711v2)

### A support must cover the task's required changes

Stack the model's outputs over the task data and linearize around the pretrained parameters:

<div class="math-display" markdown="0">
\[
f(w_0+\delta w)\approx f(w_0)+J\delta w.
\]
</div>

A selected support gives the output change <span class="math-inline" markdown="0">\(J_S\theta\)</span>. For a desired local output correction <span class="math-inline" markdown="0">\(b\)</span>, consider

<div class="math-display" markdown="0">
\[
\min_\theta\&#124;J_S\theta-b\&#124;_2^2.
\]
</div>

The optimal residual is

<div class="math-display" markdown="0">
\[
(I-J_SJ_S^\dagger)b.
\]
</div>

This makes the role of support selection concrete. The selected connections should jointly span the changes that matter to the task.

There is also a function-space interpretation. The neural tangent kernel expresses how parameter gradients combine to move a model's outputs. [Jacot, Gabriel, and Hongler, §4](https://papers.nips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf) In the local model above, the kernel available to SparseLeaf is

<div class="math-display" markdown="0">
\[
\mathcal K_S=J_SJ_S^\top
=\sum_{j\in S}J_{:j}J_{:j}^\top.
\]
</div>

For gradient flow on <span class="math-inline" markdown="0">\(\tfrac12\&#124;J_S\theta-b\&#124;_2^2\)</span>, the residual <span class="math-inline" markdown="0">\(e=J_S\theta-b\)</span> follows

<div class="math-display" markdown="0">
\[
\dot e=-\mathcal K_Se.
\]
</div>

Selecting plastic connections selects the Jacobian features that build this learning kernel. Their joint span determines which output errors can be removed; their kernel eigenvalues determine how quickly those errors decay in the local model. Support selection therefore shapes both the available changes and the dynamics of learning them.

Take four candidate coordinates with output effects

<div class="math-display" markdown="0">
\[
J=
\begin{bmatrix}
1&amp;2&amp;0&amp;1\\
0&amp;0&amp;1&amp;1
\end{bmatrix},
\qquad
b=\begin{bmatrix}1\\1\end{bmatrix}.
\]
</div>

With a budget of two coordinates, selecting the first two gives two controls along the same output direction and leaves residual <span class="math-inline" markdown="0">\((0,1)^\top\)</span>. Selecting the first and third covers the target exactly.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/05-task-coverage.png" alt="The same two-coordinate budget produces either redundant horizontal output directions or complementary horizontal and vertical directions that cover the target." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 5. Support quality depends on the combined effect of its connections. Both choices use two trainable values; the complementary support covers the two-dimensional target.*

**The value of a connection depends on the changes it enables together with the other plastic connections.**

### Allocate plasticity by its learning value

A local quadratic model makes support selection an explicit optimization problem:

<div class="math-display" markdown="0">
\[
q(\delta)=g^\top\delta+\frac12\delta^\top H\delta.
\]
</div>

For a support <span class="math-inline" markdown="0">\(S\)</span> with positive-definite <span class="math-inline" markdown="0">\(H_{SS}\)</span>, the optimal supported step and its predicted improvement are

<div class="math-display" markdown="0">
\[
\delta_S^*=-H_{SS}^{-1}g_S,
\qquad
G(S)=\frac12g_S^\top H_{SS}^{-1}g_S.
\]
</div>

The allocation problem is therefore

<div class="math-display" markdown="0">
\[
\max_{&#124;S&#124;\le K}G(S).
\]
</div>

With positive diagonal curvature, each coordinate receives score <span class="math-inline" markdown="0">\(g_i^2/(2h_i)\)</span>. For example:

| Coordinate | Gradient <span class="math-inline" markdown="0">\(g_i\)</span> | Curvature <span class="math-inline" markdown="0">\(h_i\)</span> | Optimal quadratic improvement |
|---|---:|---:|---:|
| 1 | −4 | 16 | 0.5 |
| 2 | −3 | 3 | 1.5 |
| 3 | −2 | 1 | 2.0 |
| 4 | −1 | 0.1 | 5.0 |

With one plastic connection, the largest gradient selects coordinate 1. The greatest predicted improvement selects coordinate 4. Curvature and task coverage give structure to the question of where learning should happen.

Coordinate-optimization theory gives this principle a concrete rule. Nutini and colleagues' Gauss–Southwell–Lipschitz rule selects the coordinate maximizing <span class="math-inline" markdown="0">\(&#124;g_i&#124;/\sqrt{L_i}\)</span>, where <span class="math-inline" markdown="0">\(L_i\)</span> measures coordinate-wise smoothness. For the diagonal quadratic above, <span class="math-inline" markdown="0">\(L_i=h_i\)</span>, so it selects exactly the coordinate with the greatest <span class="math-inline" markdown="0">\(g_i^2/(2h_i)\)</span> improvement. [Coordinate Descent Converges Faster with the Gauss-Southwell Rule Than Random Selection, §6.2](https://proceedings.mlr.press/v37/nutini15.html)

For neural-network adaptation, FISH Mask estimates diagonal Fisher information, selects the <span class="math-inline" markdown="0">\(K\)</span> largest entries, and reuses that fixed mask during training. It makes task sensitivity an explicit criterion for assigning plasticity. [Sung, Nair, and Raffel, Training Neural Networks with Fixed Sparse Masks](https://proceedings.neurips.cc/paper/2021/hash/cb2653f548f8709598e8b5156738cc51-Abstract.html) Liu and colleagues subsequently compare eight salience metrics and static versus dynamic masking; their results identify a simple gradient-based static mask as a strong sparse-adaptation strategy. [Refining Salience-Aware Sparse Fine-Tuning Strategies, §§3–4](https://aclanthology.org/2025.acl-long.1541/)

The experiments below use a concrete, reproducible allocation rule: within each target matrix's budget, select the largest-magnitude pretrained weights and keep that support fixed. A chunked Top-K procedure determines the coordinates; training then updates their increments and moment states.

Selection determines where learning may occur. The gradients determine how those connections change.

<span id="6-what-a-small-set-of-connections-learns"></span>

## 5. What a Small Set of Connections Learns

The experiments express learning capacity through a direct variable: the number of plastic connections.

### Supervised fine-tuning: the plasticity–quality curve

On Qwen3-1.7B, the GSM8K experiment uses completion-only supervised fine-tuning for 2,048 steps, batch size 2, and maximum sequence length 256. Validation measures completion-only negative log-likelihood on a held-out slice. Density is the proportion of coordinates selected within the targeted linear matrices.

| Trainable coordinates <span class="math-inline" markdown="0">\(K\)</span> | Density | Seeds | Validation NLL ↓ |
|---:|---:|---:|---:|
| 1,525,311 | 0.10823% | 3 | 0.4158 ± 0.0013 |
| 653,576 | 0.04638% | 5 | 0.4164 ± 0.0010 |
| 352,212 | 0.02499% | 2 | 0.4215 ± 0.0010 |
| 217,812 | 0.01546% | 3 | 0.4284 ± 0.0010 |
| 70,364 | 0.00499% | 3 | 0.4499 ± 0.0015 |
| 28,028 | 0.00199% | 3 | 0.4752 ± 0.0010 |

The ± values are sample standard deviations across seeds. The base model's NLL is 1.5935. The reported LoRA <span class="math-inline" markdown="0">\(r=8\)</span> reference has mean NLL 0.4329 across three seeds.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/06-sft-plasticity-budget.png" alt="GSM8K validation negative log-likelihood against the number of trainable coordinates, with seed standard deviations and the reported LoRA rank-eight quality reference at NLL 0.4329." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 6. The plasticity–quality curve. The left panel shows all six coordinate budgets on a logarithmic horizontal axis; the right panel expands the three largest budgets. Error bars are sample standard deviations. The horizontal reference marks LoRA <span class="math-inline" markdown="0">\(r=8\)</span> at NLL 0.4329.*

Three features of this curve explain the role of the learning budget.

First, 28,028 plastic connections produce a substantial change in task fit: NLL falls from 1.5935 to 0.4752. A pretrained model can be adapted through a very small set of connection adjustments.

Second, increasing the support expands the available adaptation. Moving from 28,028 to 352,212 coordinates lowers NLL from 0.4752 to 0.4215.

Third, the curve flattens. Reducing the budget from 1,525,311 to 653,576 removes approximately 57% of the trainable values, while the reported mean NLL changes from 0.4158 to 0.4164.

The curve turns a parameter count into a learning question: how much task improvement does each additional allocation of plasticity buy?

### Reinforcement learning: the same connections receive reward

The MATH GSPO runs use the same Qwen3-1.7B base, fixed coordinate supports, and 120 training steps. Both runs use seed 126 and evaluate with greedy decoding on the same 200-question slice.

| Target coordinate density | Plastic connections | Before training | After training | Change |
|---:|---:|---:|---:|---:|
| 0.030% | 422,688 | 53.0% | 55.0% | +2.0 percentage points |
| 0.300% | 4,227,720 | 53.0% | 59.5% | +6.5 percentage points |

The recorded runs keep base-weight gradients at zero and maintain optimizer moments on the selected increments. The learning objective supplies reward-based signals to the same coordinate parameterization used for supervised learning.

<img src="/images/blog/sparseleaf-synaptic-sparse-fine-tuning/07-rl-plasticity.png" alt="Before-and-after MATH evaluation scores for two coordinate densities, alongside their recorded per-step rewards and trailing twenty-step averages." width="2560" height="1280" loading="lazy" decoding="async">

*Figure 7. MATH GSPO records. Left: greedy accuracy on the shared 200-question evaluation slice. Right: recorded total reward at each training step, with a trailing 20-step mean. Each density uses seed 126.*

Together, the SFT and RL results demonstrate a common learning interface. Supervised errors and reward signals both reach a small collection of independent connections, while the complete pretrained network carries out the computation.

<span id="the-engineering-conveniences-of-coordinate-sparsity"></span>

## 6. The Engineering Conveniences of Coordinate Sparsity

The parameterization also determines what engineers must store, schedule, differentiate, and move. LoRA's trainable-parameter percentage compresses four different bills into one attractive number: arithmetic, memory traffic, batching, and persistent state. Coordinate sparsity changes the objects behind each bill: learned projection pairs become indexed connection values, and the support becomes reusable execution metadata.

### Where the Memory Savings Actually Happen

The pretrained network supplies the computation. Only the plastic connections carry trainable values, parameter gradients, and optimizer history.

Let <span class="math-inline" markdown="0">\(N\)</span> be the number of base-weight values and <span class="math-inline" markdown="0">\(K\)</span> the number of selected coordinates. The storage representation follows the parameterization directly:

<div class="math-display" markdown="0">
\[
w=w_0+P_S^\top\theta,
\qquad
w_0\in\mathbb R^N,\quad
\theta\in\mathbb R^K.
\]
</div>

The frozen base is stored once. The increment, its gradient, and its optimizer state are separate, coordinate-sized tensors. This places the savings at specific points in the training lifecycle.

#### Weights: reuse the base, allocate precision to the increments

The base retains its low-precision computation weights. A trainable vector holds the selected increments, and an FP32 master vector accumulates their updates before conversion to the computation dtype. Mixed-precision training uses master weights to preserve small updates across optimization steps; SparseLeaf allocates that precision to the connections that can change. The master copy therefore shrinks from <span class="math-inline" markdown="0">\(N\)</span> values to <span class="math-inline" markdown="0">\(K\)</span> values. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

Multiple tasks can reuse the same immutable base and maintain independent increment vectors. Saving an adapter records its support and values, while the base remains a separately stored asset. Once a receiver has the same base and support, a new adapter version can be transmitted as a replacement value vector. The savings appear in trainable weight storage, task-specific checkpoints, and version payloads.

#### Weight gradients: write the selected results directly

For <span class="math-inline" markdown="0">\(Y=XW^\top\)</span> and <span class="math-inline" markdown="0">\(D=\partial\mathcal L/\partial Y\)</span>, full-parameter training forms <span class="math-inline" markdown="0">\(G_W=D^\top X\)</span>, with one gradient value per weight. SparseLeaf's backward operator instead writes

<div class="math-display" markdown="0">
\[
(g_\theta)_k
=\sum_t D_{t,i_k}X_{t,j_k},
\qquad
g_\theta\in\mathbb R^K.
\]
</div>

Each selected connection contributes one dot product across token positions. The operator writes those results directly into a length-<span class="math-inline" markdown="0">\(K\)</span> gradient buffer. The frozen base has no parameter-gradient buffer; the increment vector receives the accumulated gradient. This matches autograd's distinction between differentiating through an operation and accumulating a gradient for one of its inputs. [PyTorch Autograd: setting requires_grad](https://docs.pytorch.org/docs/2.14/notes/autograd.html#setting-requires-grad)

The saving begins at gradient generation: selected dot products replace the full weight-gradient multiplication, and <span class="math-inline" markdown="0">\(K\)</span> output writes replace <span class="math-inline" markdown="0">\(N\)</span>. Microbatch accumulation then operates on the same <span class="math-inline" markdown="0">\(K\)</span> entries. Data-parallel replicas with an aligned fixed support synchronize those entries in a common coordinate order. The layer's full input-gradient path continues to carry errors to earlier layers; the backward section below separates that computation from parameter-gradient generation.

#### Optimizer state: keep history only where learning can happen

Adam maintains a first moment and a second moment for each trainable value. With a sparse leaf, both histories have length <span class="math-inline" markdown="0">\(K\)</span>. The optimizer creates the master vector and moment vectors in the leaf's shape, updates them from its gradient, and writes back the new increment values. State allocation, per-step reads and writes, and optimizer checkpoints all follow the selected coordinates.

Using BF16 computation values and parameter gradients, an FP32 master copy, and two FP32 Adam moments gives the following logical tensor accounting. ZeRO uses this same separation of weights, gradients, master weights, and moments to explain the memory cost of mixed-precision Adam. [ZeRO, §3.1](https://arxiv.org/html/1910.02054v3)

| Tensor | Full-parameter training | SparseLeaf |
|---|---:|---:|
| Complete computation weights / frozen base | <span class="math-inline" markdown="0">\(2N\)</span> bytes | <span class="math-inline" markdown="0">\(2N\)</span> bytes |
| Separate BF16 increments | — | <span class="math-inline" markdown="0">\(2K\)</span> bytes |
| Parameter gradients | <span class="math-inline" markdown="0">\(2N\)</span> bytes | <span class="math-inline" markdown="0">\(2K\)</span> bytes |
| FP32 master weights | <span class="math-inline" markdown="0">\(4N\)</span> bytes | <span class="math-inline" markdown="0">\(4K\)</span> bytes |
| Adam first moment | <span class="math-inline" markdown="0">\(4N\)</span> bytes | <span class="math-inline" markdown="0">\(4K\)</span> bytes |
| Adam second moment | <span class="math-inline" markdown="0">\(4N\)</span> bytes | <span class="math-inline" markdown="0">\(4K\)</span> bytes |
| **Core tensor total** | **<span class="math-inline" markdown="0">\(16N\)</span> bytes** | **<span class="math-inline" markdown="0">\(2N+16K\)</span> bytes** |

The full memory ledger adds coordinate metadata, activation storage, and operator workspace. An FP32 microbatch-accumulation buffer contributes another <span class="math-inline" markdown="0">\(4K\)</span> bytes while resident. In the core total above, the BF16 parameter-gradient buffer contributes <span class="math-inline" markdown="0">\(2K\)</span> bytes; accumulation and optimizer staging have their own lifetimes.

Return to the opening <span class="math-inline" markdown="0">\(4096\times4096\)</span> matrix with <span class="math-inline" markdown="0">\(K=4096\)</span> selected connections. Full-parameter training uses **256 MiB** for the core tensors in the table. SparseLeaf uses **32 MiB for the base plus 64 KiB for the increment training state**: 8 KiB of increments, 8 KiB of gradients, 16 KiB of master weights, and 32 KiB for the two moments. Flat INT32 coordinates add 16 KiB; CSR/CSC execution metadata is a separate allocation.

For standard LoRA with <span class="math-inline" markdown="0">\(P=\sum_\ell r_\ell(m_\ell+n_\ell)\)</span> factor values, the same accounting gives

<div class="math-display" markdown="0">
\[
M_{\mathrm{LoRA,core}}=2N+16P,
\qquad
M_{\mathrm{SparseLeaf,core}}=2N+16K.
\]
</div>

Coordinate metadata adds <span class="math-inline" markdown="0">\(B_S\)</span> to SparseLeaf's storage. Both representations reuse a frozen base. SparseLeaf makes the learning-state budget divisible into individual connections: removing one coordinate removes its increment, gradient, master value, and both moment entries. Every retained coordinate keeps an independent update direction. The memory budget follows the number of plastic connections while their combined update remains free to have high rank.

### From skinny GEMMs to indexed accumulation

For <span class="math-inline" markdown="0">\(T\)</span> token rows, an unmerged LoRA layer computes

<div class="math-display" markdown="0">
\[
Z=XA^\top,\qquad
Y=XW_0^\top+ZB^\top.
\]
</div>

The adapter adds two dependent, skinny matrix multiplications. Reducing <span class="math-inline" markdown="0">\(r\)</span> narrows the first projection's output dimension and the second projection's reduction dimension. It shrinks their arithmetic and the work available to amortize scheduling and memory access. Each adapted gate, up, and down projection in an MoE expert carries this pair: six logical adapter multiplications per expert forward pass.

NVIDIA reports exactly this shape problem: even its heterogeneous batched GEMM underutilized the GPU on the first LoRA projection. Its implementation used split-K parallelism and an additional reduction kernel to improve utilization. A small factor matrix still required extra scheduling and reduction work. [NVIDIA NIM implementation notes](https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/)

The traffic arithmetic is revealing. For <span class="math-inline" markdown="0">\(q\)</span> token rows, factors of shapes <span class="math-inline" markdown="0">\(r\times n\)</span> and <span class="math-inline" markdown="0">\(m\times r\)</span>, and <span class="math-inline" markdown="0">\(b\)</span> bytes per value, count one read of each factor and input, and one output write:

<div class="math-display" markdown="0">
\[
I_{\mathrm{traffic}}
=\frac{2qr(m+n)}
{b\,[r(m+n)+q(n+m)]}
=\frac{2qr}{b(q+r)}
\quad\text{FLOPs per byte}.
\]
</div>

At one token and BF16, this model gives <span class="math-inline" markdown="0">\(I_{\mathrm{traffic}}=r/(r+1)&lt;1\)</span>. Intermediate tensors and output accumulation add traffic. The GPU executes shapes and memory transfers, not parameter-efficiency percentages.

SparseLeaf replaces the two learned contractions with a direct connection-wise operation:

<div class="math-display" markdown="0">
\[
(\Delta y_t)_i
=\sum_{k:i_k=i}\theta_k x_{t,j_k}.
\]
</div>

A row-grouped kernel gathers the selected inputs, multiplies by their increment values, and accumulates directly into the corresponding outputs. The adapter has no rank-<span class="math-inline" markdown="0">\(r\)</span> intermediate <span class="math-inline" markdown="0">\(Z\)</span> to produce and consume. Its work is parallelized over tokens and coordinate blocks, instead of organized around two skinny learned projections.

Writing <span class="math-inline" markdown="0">\(P=r(m+n)\)</span>, the forward arithmetic becomes <span class="math-inline" markdown="0">\(2TK\)</span> connection FLOPs in place of <span class="math-inline" markdown="0">\(2TP\)</span> factor FLOPs. Reducing <span class="math-inline" markdown="0">\(K\)</span> removes connections from the work list. It does not squeeze a learned projection into an increasingly narrow matrix shape. This targets both the amount of adapter work and its two-stage dependency.

### From factor gradients to direct connection gradients

Let <span class="math-inline" markdown="0">\(D=\partial\mathcal L/\partial Y\)</span> and reuse <span class="math-inline" markdown="0">\(Z=XA^\top\)</span>. LoRA's backward pass contains

<div class="math-display" markdown="0">
\[
\begin{aligned}
C&amp;=DB,&amp;
\nabla_B\mathcal L&amp;=D^\top Z,\\
\nabla_A\mathcal L&amp;=C^\top X,&amp;
\nabla_X\mathcal L&amp;=DW_0+CA.
\end{aligned}
\]
</div>

Freezing <span class="math-inline" markdown="0">\(W_0\)</span> removes its parameter gradient. It does not remove the dense multiplication <span class="math-inline" markdown="0">\(DW_0\)</span> that propagates learning signals to earlier layers. The adapter additionally contributes four skinny backward multiplications. CE-LoRA identifies this dense activation-gradient multiplication as the central computational bottleneck. [CE-LoRA, §2.2](https://arxiv.org/html/2502.01378)

Count the forward and backward matrix multiplications of an interior linear layer, with both factors trained, unit adapter scaling, and two FLOPs per multiply-add:

<div class="math-display" markdown="0">
\[
F_{\mathrm{full}}=6Tmn,\qquad
F_{\mathrm{LoRA}}=4Tmn+6Tr(m+n).
\]
</div>

Thus

<div class="math-display" markdown="0">
\[
\frac{F_{\mathrm{LoRA}}}{F_{\mathrm{full}}}
=\frac23+\frac{r(m+n)}{mn}.
\]
</div>

An adapter with 0.1% of the layer's parameters still uses approximately 66.77% of its full-training matrix-multiplication FLOPs. That is the arithmetic before accounting for the execution efficiency of the added skinny operations.

The activation bill is equally explicit: <span class="math-inline" markdown="0">\(\nabla_A\mathcal L=C^\top X\)</span> needs the full-width input <span class="math-inline" markdown="0">\(X\)</span>. Making <span class="math-inline" markdown="0">\(r\)</span> smaller does not shrink <span class="math-inline" markdown="0">\(X\)</span>; it must be retained or recomputed. LoRA-FA freezes the input projection precisely to remove this storage requirement. [LoRA-FA, §§2.2–3.1](https://arxiv.org/html/2308.03303v1)

Parameter gradients, activation gradients, and saved activations are separate objects. Counting only the first obscures the training bottleneck.

SparseLeaf builds the parameter-gradient vector directly from the selected connections:

<div class="math-display" markdown="0">
\[
\frac{\partial\mathcal L}{\partial\theta_k}
=\sum_t D_{t,i_k}X_{t,j_k}.
\]
</div>

The kernel reduces along the token dimension and writes one result per coordinate. It produces neither a full weight-gradient matrix nor the factor-gradient chain through <span class="math-inline" markdown="0">\(DB\)</span>. Gradient accumulation, Adam updates, and parameter-gradient synchronization operate on these <span class="math-inline" markdown="0">\(K\)</span> values.

The full activation gradient is

<div class="math-display" markdown="0">
\[
\nabla_X\mathcal L
=DW_0+\sum_k\theta_kD_{:,i_k}e_{j_k}^\top.
\]
</div>

This preserves complete error transport through the pretrained network and makes the adapter correction a direct sparse accumulation. Under the same linear-layer arithmetic accounting,

<div class="math-display" markdown="0">
\[
F_{\mathrm{SparseLeaf}}=4Tmn+6TK.
\]
</div>

The shared base computation is explicit; the adapter's forward, parameter-gradient, and input-gradient work scales with the selected connections. The <span class="math-inline" markdown="0">\(6TP\)</span> factor-side work has become <span class="math-inline" markdown="0">\(6TK\)</span> connection-side work.

Activation retention also becomes an explicit support-layout problem. Define <span class="math-inline" markdown="0">\(J(S)=\{j_k:k=1,\ldots,K\}\)</span>. A support-aware backward pass saves <span class="math-inline" markdown="0">\(X_{:,J(S)}\)</span> for the selected parameter dot products, requiring <span class="math-inline" markdown="0">\(bT&#124;J(S)&#124;\)</span> input-activation bytes. The engineer can optimize column coverage and reuse directly. Rank reduction offers no equivalent control over the full-width <span class="math-inline" markdown="0">\(X\)</span> needed to train <span class="math-inline" markdown="0">\(A\)</span>.

### From tenant–expert fragments to a shared coordinate schedule

With <span class="math-inline" markdown="0">\(E\)</span> routed experts, top-<span class="math-inline" markdown="0">\(k\)</span> routing, <span class="math-inline" markdown="0">\(U\)</span> tenants, and <span class="math-inline" markdown="0">\(T\)</span> token rows, balanced routing gives average group sizes

<div class="math-display" markdown="0">
\[
q_{\mathrm{expert}}=\frac{Tk}{E},
\qquad
q_{\mathrm{tenant,expert}}=\frac{Tk}{UE}.
\]
</div>

The base expert can process tokens from different tenants together. Independent tenant adapters have different factors, so their reuse groups are tenant–expert pairs.

Take <span class="math-inline" markdown="0">\(T=512\)</span>, <span class="math-inline" markdown="0">\(k=2\)</span>, <span class="math-inline" markdown="0">\(E=64\)</span>, and <span class="math-inline" markdown="0">\(U=16\)</span>. The base has 16 routed tokens per expert on average. The adapters have one routed token per tenant–expert cell on average. The apparent 512-token batch has become a grid of tiny adapter problems.

Grouped and fused kernels pack these fragments into fewer launches. They do not make different tenants reuse the same factor values. Increasing the expert count expands the collection of adapter states while reducing the token reuse available to each expert. Increasing the tenant count repeats that pressure.

Coordinate sparsity separates the execution structure from the tenant's learned values. For a shared support <span class="math-inline" markdown="0">\(S_e=\{(i_{e,k},j_{e,k})\}_{k=1}^{K_e}\)</span> in one expert projection,

<div class="math-display" markdown="0">
\[
(\Delta y_{t,e})_i
=\sum_{\substack{1\le k\le K_e\\i_{e,k}=i}}
\theta_{u(t),e,k}\,(x_{t,e})_{j_{e,k}}.
\]
</div>

The indices and output-reduction pattern are identical across tenants. The tenant identifier selects a value vector, not a new pair of learned matrix contractions. This gives a fused kernel a scheduling key of **expert plus coordinate block**: traverse the expert's token batch using one coordinate schedule and fetch each token's tenant-specific values. In the example above, that schedule spans the expert's 16 tokens instead of creating a separate projection chain for every one-token tenant–expert cell.

The same locality determines device placement. Put each increment on the device that owns its base-weight coordinate. Expert-local corrections join the existing expert output; tensor-parallel partial corrections join the base layer's existing output reduction. There is no additional rank-dimensional intermediate to assemble across shards.

### From adapter cache pressure to sparse value tables

Merging an adapter materializes a tenant-specific dense weight <span class="math-inline" markdown="0">\(W_0+B_uA_u\)</span>. Different tenants then need different merged weights. Keeping one shared base instead leaves tenant-specific adapter computation on the serving path.

This is why multi-LoRA serving requires its own systems machinery. Punica introduces segmented gather matrix-vector multiplication to batch adapter work and group requests using the same factors. S-LoRA manages adapter weights and KV caches in one paged memory pool, adds heterogeneous kernels, and prefetches adapters. Adapter residency consumes memory that could otherwise hold longer contexts or more concurrent requests. Clustering requests by adapter improves reuse by changing who gets served together—and when. [Punica, §§3–4](https://arxiv.org/abs/2310.18547), [S-LoRA, §§4–5](https://arxiv.org/abs/2311.03285)

The cost is a chain: more adapter identities mean more distinct weights, weaker per-adapter reuse, and more pressure on cache capacity and scheduling. Loading, eviction, transfer, and batching become part of the customization bill.

With a fixed shared coordinate support, tenant-specific storage is a value table. For one projection, let <span class="math-inline" markdown="0">\(B_S\)</span> be the shared coordinate-metadata size and <span class="math-inline" markdown="0">\(b_w\)</span> the number of bytes per learned value. Adapter storage for <span class="math-inline" markdown="0">\(U\)</span> resident tenants is

<div class="math-display" markdown="0">
\[
M_{\mathrm{LoRA}}=Ub_wP,
\qquad
M_{\mathrm{coordinate}}=B_S+Ub_wK.
\]
</div>

Each new tenant adds <span class="math-inline" markdown="0">\(K\)</span> values; the index table and reduction layout are reused. Smaller coordinate budgets therefore translate directly into smaller resident payloads, more room for KV caches, and fewer bytes on cache misses. Version updates use that same coordinate map, transmitting replacement increment values without rebuilding a factorized weight change.

For a materialized model version, patching touches the selected coordinates. A low-rank merge computes <span class="math-inline" markdown="0">\(BA\)</span> and writes a generally dense change; a coordinate patch writes <span class="math-inline" markdown="0">\(K\)</span> locations. SHiRA demonstrates this sparse-write approach through indexed adapter switching. [Sparse High Rank Adapters, §3.2](https://arxiv.org/html/2406.13175)

### From the rank-one floor to one-connection budgets

Rank is an integer. For a fixed collection <span class="math-inline" markdown="0">\(\mathcal T\)</span> of independently adapted matrices, the smallest nonzero standard LoRA allocation to every target is

<div class="math-display" markdown="0">
\[
P_{\min}(\mathcal T)
=\sum_{\ell\in\mathcal T}(m_\ell+n_\ell).
\]
</div>

The next smaller rank is zero: removing that matrix's adapter. LoRA cannot spend three independent scalar parameters on a projection whose rank-one factorization costs thousands.

Consider Kimi K2's routed experts. Its architecture has 61 layers, one dense layer, 384 routed experts per MoE layer, model width 7,168, and expert width 2,048. That gives 60 MoE layers. [Kimi K2 technical report, §2.3](https://arxiv.org/html/2507.20534)

Allocate independent rank-<span class="math-inline" markdown="0">\(r\)</span> LoRA factors to the gate, up, and down projections of every routed expert. The resulting factor count is

<div class="math-display" markdown="0">
\[
\begin{aligned}
P_r
&amp;=3\times60\times384\times r(7{,}168+2{,}048)\\
&amp;=637{,}009{,}920\,r.
\end{aligned}
\]
</div>

The model activates eight routed experts per token, but the complete adapter contains factors for all 384. Token-level conditional computation does not shrink the all-expert checkpoint or its eventual optimizer state.

This is the rank-one trap: 637 million trainable values, yet each adapted projection is still restricted to one rank-one change. The granularity is simultaneously too expensive for tiny budgets and too restrictive for independent connection updates.

Coordinate sparsity breaks this floor while retaining the same projection coverage. Give every one of those 69,120 expert projections 16 selected entries in distinct rows and columns:

<div class="math-display" markdown="0">
\[
K=3\times60\times384\times16=1{,}105{,}920.
\]
</div>

That is **576 times fewer trainable values than rank-one LoRA**, and each projection's update can attain rank 16. The budget is spent on independent connection changes across the same target matrices.

| Allocation per routed-expert projection | Trainable values | BF16 learned values | Two FP32 Adam moments |
|---|---:|---:|---:|
| LoRA rank 1 | 637,009,920 | 1.274 GB | 5.096 GB |
| LoRA rank 8 | 5,096,079,360 | 10.192 GB | 40.769 GB |
| SparseLeaf: 16 coordinates | 1,105,920 | 2.212 MB | 8.847 MB |

*Derived allocation and storage accounting for the same 69,120 projections. Decimal units; learned values use two bytes each and the moment pair uses eight. Shared coordinate metadata is stored separately.*

With BF16 values and gradients, an FP32 master copy, and two FP32 Adam moments, the training-state totals are <span class="math-inline" markdown="0">\(16P_1=10.192\)</span> GB for rank-one LoRA and <span class="math-inline" markdown="0">\(16K=17.695\)</span> MB for this coordinate allocation. For 100 resident tenants, their BF16 value payloads are 127.402 GB and 221.184 MB, respectively.

One extra LoRA rank purchases <span class="math-inline" markdown="0">\(m+n\)</span> factor values in a target matrix. One extra SparseLeaf coordinate purchases one connection. That granularity lets an engineer allocate a global budget across layers and experts without treating a rank-one adapter as an indivisible minimum purchase.

The engineering correspondence is direct: indexed accumulation replaces skinny projection chains; shared supports provide reusable multi-tenant schedules; selected dot products produce the learning gradients; and coordinate budgets control resident state and update payloads. These conveniences follow from making the connections themselves plastic.

## From Parameter Compression to Synaptic Plasticity

The 4,096-entry example at the beginning captures the central idea: an update can be small in coordinates and large in rank.

The task construction turns that idea into a strict loss separation. The optimizer shows how the selected connections receive their native gradients. The allocation problem asks which connections are worth making plastic. The experiments show what those connections learn under supervision and reward.

Together, they suggest a different organizing principle for parameter-efficient learning: **spend the budget on choosing plastic connections, and preserve their freedom to create independent changes.**

SparseLeaf makes the support of learning an explicit part of the optimization problem. It preserves the pretrained network's computation, gives selected connections independent increments, and lets those increments form high-rank updates.

The important unit is the plastic connection. Its location determines where a change enters the network. Its gradient determines what the task asks it to do. Its optimizer state records how that request develops over time. Many such local changes can combine into a high-rank transformation.

**Dense capacity. Sparse plasticity. High-rank updates.**

## References

1. Hu, E. J., et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685; ICLR 2022.
2. Eckart, C., and Young, G. (1936). [The Approximation of One Matrix by Another of Lower Rank](https://www.cambridge.org/core/journals/psychometrika/article/approximation-of-one-matrix-by-another-of-lower-rank/B29672E1EDD0FA1B7611D4DFAFC321B3). *Psychometrika*, 1, 211–218.
3. Zeng, Y., and Lee, K. (2024). [The Expressive Power of Low-Rank Adaptation](https://arxiv.org/html/2310.17513). ICLR.
4. Aghajanyan, A., Gupta, S., and Zettlemoyer, L. (2021). [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://aclanthology.org/2021.acl-long.568/). ACL-IJCNLP.
5. Bhardwaj, K., et al. (2024; revised 2025). [Sparse High Rank Adapters](https://arxiv.org/html/2406.13175). arXiv:2406.13175.
6. Liu, X., et al. (2025). [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](https://aclanthology.org/2025.acl-long.1541/). ACL.
7. Hayou, S., Ghosh, N., and Yu, B. (2024). [The Impact of Initialization on LoRA Finetuning Dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html). NeurIPS.
8. Hao, Y., Cao, Y., and Mou, L. (2024). [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://proceedings.mlr.press/v235/hao24a.html). ICML, PMLR 235, 17554–17571.
9. Kalajdzievski, D. (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/html/2312.03732). arXiv:2312.03732.
10. Hayou, S., Ghosh, N., and Yu, B. (2024). [LoRA+: Efficient Low Rank Adaptation of Large Models](https://proceedings.mlr.press/v235/hayou24a.html). ICML, PMLR 235, 17783–17806.
11. Yen, J.-N., et al. (2025). [LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](https://arxiv.org/html/2410.20625). ICLR; arXiv:2410.20625.
12. Wang, Z., Liang, J., He, R., Wang, Z., and Tan, T. (2025). [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/html/2407.18242). ICLR; arXiv:2407.18242.
13. Wang, S., Yu, L., and Li, J. (2024). [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/html/2407.05000). arXiv:2407.05000; NeurIPS.
14. Meng, F., Wang, Z., and Zhang, M. (2024; revised 2025). [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://arxiv.org/html/2404.02948). arXiv:2404.02948; NeurIPS 2024.
15. Paischer, F., Hauzenberger, L., Schmied, T., Alkin, B., Deisenroth, M. P., and Hochreiter, S. (2025). [Parameter Efficient Fine-tuning via Explained Variance Adaptation](https://arxiv.org/html/2410.07170). NeurIPS; arXiv:2410.07170.
16. Shuttleworth, R., Andreas, J., Torralba, A., and Sharma, P. (2024; revised 2025). [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/html/2410.21228v3). arXiv:2410.21228, version 3.
17. Matsuzaki, M., Honkura, N., Ellis-Davies, G. C. R., and Kasai, H. (2004). [Structural basis of long-term potentiation in single dendritic spines](https://pmc.ncbi.nlm.nih.gov/articles/PMC4158816/). *Nature*, 429, 761–766.
18. Guo, D., Rush, A. M., and Kim, Y. (2021). [Parameter-Efficient Transfer Learning with Diff Pruning](https://aclanthology.org/2021.acl-long.378/). ACL-IJCNLP.
19. Rios, J., Dognin, P., Luss, R., and Natesan Ramamurthy, K. (2025). [Sparsity May Be All You Need: Sparse Random Parameter Adaptation](https://aclanthology.org/2025.findings-emnlp.1013/). Findings of EMNLP.
20. Kingma, D. P., and Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR.
21. Duchi, J., Hazan, E., and Singer, Y. (2011). [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/v12/duchi11a.html). *Journal of Machine Learning Research*, 12, 2121–2159.
22. Williams, R. J. (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). *Machine Learning*, 8, 229–256.
23. Mukherjee, S., Yuan, L., Hakkani-Tür, D., and Peng, H. (2025). [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/html/2505.11711v2). arXiv:2505.11711, version 2.
24. Jacot, A., Gabriel, F., and Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://papers.nips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf). NeurIPS.
25. Nutini, J., Schmidt, M., Laradji, I., Friedlander, M., and Koepke, H. (2015). [Coordinate Descent Converges Faster with the Gauss-Southwell Rule Than Random Selection](https://proceedings.mlr.press/v37/nutini15.html). ICML.
26. Sung, Y.-L., Nair, V., and Raffel, C. (2021). [Training Neural Networks with Fixed Sparse Masks](https://proceedings.neurips.cc/paper/2021/hash/cb2653f548f8709598e8b5156738cc51-Abstract.html). NeurIPS.
27. Micikevicius, P., et al. (2018). [Mixed Precision Training](https://arxiv.org/abs/1710.03740). ICLR.
28. PyTorch Contributors (2026). [PyTorch Autograd: setting requires_grad](https://docs.pytorch.org/docs/2.14/notes/autograd.html#setting-requires-grad). PyTorch 2.14 documentation.
29. Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. (2020). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/html/1910.02054v3). arXiv:1910.02054, version 3.
30. Verma, S., et al. (2024). [Seamlessly Deploying a Swarm of LoRA Adapters with NVIDIA NIM](https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim/). NVIDIA Technical Blog, June 7.
31. Chen, G., He, Y., Hu, Y., Yuan, K., and Yuan, B. (2025). [CE-LoRA: Computation-Efficient LoRA Fine-Tuning for Language Models](https://arxiv.org/html/2502.01378). arXiv:2502.01378.
32. Zhang, L., Zhang, L., Shi, S., Chu, X., and Li, B. (2023). [LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning](https://arxiv.org/html/2308.03303v1). arXiv:2308.03303, version 1.
33. Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., and Krishnamurthy, A. (2023). [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547). arXiv:2310.18547; MLSys 2024.
34. Sheng, Y., et al. (2023; revised 2024). [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285). arXiv:2311.03285; MLSys 2024.
35. Kimi Team (2025; revised 2026). [Kimi K2: Open Agentic Intelligence](https://arxiv.org/html/2507.20534). arXiv:2507.20534.
