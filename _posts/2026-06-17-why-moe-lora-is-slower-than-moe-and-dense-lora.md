---
layout: post
title: "Why MoE-LoRA Is Slower Than MoE and Dense-LoRA: A Mathematical View"
date: 2026-06-17T00:00:00.000+08:00
categories: [Blog]
tags: [moe, lora, llm-inference, systems]
math: true
excerpt: "MoE-LoRA is slow not because LoRA adds many FLOPs, but because sparse routing turns low-rank updates into many tiny, dynamic, low-reuse matrix multiplications."
---

MoE-LoRA is slow for a slightly counterintuitive reason. LoRA itself is
FLOP-light, but MoE-LoRA changes the serving problem from a few large, regular
matrix multiplications into many small, dynamic, low-rank, block-sparse matrix
multiplications.

The key operator is:

<div class="math-display">
\[
Y_{\Delta}
=
\sum_{e,a}
D_{e,a} X A_{e,a} B_{e,a}.
\]
</div>

Here \(X \in \mathbb{R}^{N \times d}\) is the token matrix, \(e\) indexes
experts, \(a\) indexes LoRA adapters, and \(D_{e,a}\) selects the tokens routed
to expert \(e\) while using adapter \(a\). Each LoRA update is factorized as
\(A_{e,a}B_{e,a}\), with \(A_{e,a} \in \mathbb{R}^{d \times r}\),
\(B_{e,a} \in \mathbb{R}^{r \times h}\), and \(r \ll d,h\).

This is the core point:

<div class="math-display">
\[
\boxed{
\text{MoE-LoRA is a dynamic token-block-sparse low-rank matrix multiplication.}
}
\]
</div>

That structure is what makes it slower than both base MoE and dense-LoRA.

## The Three Operators

A dense linear layer is simple:

<div class="math-display">
\[
Y = XW,
\qquad
X \in \mathbb{R}^{N \times d},
\quad
W \in \mathbb{R}^{d \times h}.
\]
</div>

Its cost is approximately:

<div class="math-display">
\[
\text{FLOPs}_{\text{dense}} \approx 2Ndh.
\]
</div>

This is exactly the kind of large dense GEMM that GPUs handle well. The weight
matrix \(W\) is reused across all \(N\) tokens, memory access is regular, and
Tensor Cores can be kept busy.

Dense-LoRA adds a low-rank update:

<div class="math-display">
\[
Y = X(W + AB) = XW + XAB.
\]
</div>

The LoRA path can be computed as:

<div class="math-display">
\[
XAB = (XA)B,
\]
</div>

with cost:

<div class="math-display">
\[
\text{FLOPs}_{\text{dense-LoRA}}
\approx
2Ndr + 2Nrh
=
2Nr(d+h).
\]
</div>

Since \(r \ll d,h\), this adds few FLOPs relative to \(XW\). More importantly,
all tokens share the same \(A\) and \(B\), so the adapter path is globally
batched.

Base MoE has a different structure. If \(D_e\) selects the tokens routed to
expert \(e\), then:

<div class="math-display">
\[
Y_{\text{MoE}}
=
\sum_e D_e X W_e.
\]
</div>

In implementation, tokens are compacted by expert:

<div class="math-display">
\[
Y_e = X_e W_e,
\qquad
X_e \in \mathbb{R}^{n_e \times d}.
\]
</div>

This is harder than a dense layer because \(n_e\) is dynamic and uneven. But
each expert still performs a full-rank dense GEMM:

<div class="math-display">
\[
[n_e,d] \times [d,h].
\]
</div>

If \(n_e\) is not too small, the expert weight \(W_e\) is still reused across
enough tokens to make the computation reasonably efficient.

## What MoE-LoRA Changes

With LoRA on top of MoE, the delta path becomes:

<div class="math-display">
\[
Y_{\Delta}^{\text{MoE-LoRA}}
=
\sum_{e,a}
D_{e,a} X A_{e,a} B_{e,a}.
\]
</div>

Let \(g=(e,a)\) be one expert-adapter group. After compaction, each active group
computes:

<div class="math-display">
\[
Y_g = X_g A_g B_g,
\qquad
X_g \in \mathbb{R}^{n_g \times d}.
\]
</div>

The matrices \(A_gB_g\) are low-rank, since:

<div class="math-display">
\[
\operatorname{rank}(A_gB_g) \le r.
\]
</div>

But the tokens are sparse and dynamically selected. So MoE-LoRA is not simply
"LoRA applied to MoE." It is many token blocks, each using a different
low-rank update, followed by scatter-add back into the original token order.

That is where the slowdown comes from.

## The Bottleneck Is Fragmentation

Base MoE groups tokens by expert. MoE-LoRA groups tokens by expert and adapter.

Base MoE has token counts:

<div class="math-display">
\[
n_e.
\]
</div>

MoE-LoRA has token counts:

<div class="math-display">
\[
n_{e,a},
\qquad
\sum_a n_{e,a} = n_e.
\]
</div>

If a batch mixes many adapters, each expert block is split into smaller
expert-adapter blocks. With \(A_{\text{active}}\) active adapters, a rough
average is:

<div class="math-display">
\[
n_{e,a} \approx \frac{n_e}{A_{\text{active}}}.
\]
</div>

So the reuse hierarchy becomes:

<div class="math-display">
\[
N
\rightarrow
n_e
\rightarrow
n_{e,a}.
\]
</div>

Dense-LoRA reuses \(A,B\) across \(N\) tokens. Base MoE reuses \(W_e\) across
\(n_e\) tokens. MoE-LoRA only reuses \(A_{e,a},B_{e,a}\) across \(n_{e,a}\)
tokens.

Usually:

<div class="math-display">
\[
n_{e,a} \ll n_e \ll N.
\]
</div>

This is the central performance problem. MoE-LoRA has the worst reuse pattern.

## Skinny GEMMs Are Not Automatically Fast

For one expert-adapter group, MoE-LoRA computes:

<div class="math-display">
\[
[n_g,d] \times [d,r]
\quad\text{then}\quad
[n_g,r] \times [r,h].
\]
</div>

Because \(r\) is small, these are skinny GEMMs. For example, with
\(d=4096\), \(h=11008\), and \(r=16\), the two LoRA projections are:

<div class="math-display">
\[
[n_g,4096] \times [4096,16],
\qquad
[n_g,16] \times [16,11008].
\]
</div>

If \(n_g\) is also small, say \(1,2,4,\) or \(8\), the operation has few FLOPs
but poor hardware utilization. The GPU is not slow because there is too much
math. It is slow because the math is too fragmented to keep the hardware fed.

This is the paradox:

<div class="math-display">
\[
\boxed{
\text{LoRA reduces FLOPs, but may reduce hardware efficiency even more.}
}
\]
</div>

## Arithmetic Intensity Drops

For one group:

<div class="math-display">
\[
Y_g = X_g A_g B_g.
\]
</div>

The FLOPs are roughly:

<div class="math-display">
\[
\text{FLOPs}_g
\approx
2n_gdr + 2n_grh
=
2n_gr(d+h).
\]
</div>

Ignoring dtype size, the data movement includes \(X_g\), \(A_g\), \(B_g\), and
\(Y_g\). A rough arithmetic-intensity model is:

<div class="math-display">
\[
\text{AI}_g
\approx
\frac{
2n_gr(d+h)
}{
n_gd + dr + rh + n_gh
}.
\]
</div>

The important variable is \(n_g\). When \(n_g\) is small, the weight-read terms
\(dr+rh\) cannot be amortized across many tokens. Arithmetic intensity falls,
and the roofline model says performance becomes memory-bandwidth-bound:

<div class="math-display">
\[
\text{attainable performance}
\le
\min(
\text{peak FLOPs},
\text{memory bandwidth} \times \text{AI}
).
\]
</div>

So MoE-LoRA can have low FLOP count and still run slowly.

## Why It Is Slower Than Base MoE

Base MoE computes:

<div class="math-display">
\[
\sum_e D_e X W_e.
\]
</div>

MoE-LoRA adds:

<div class="math-display">
\[
\sum_{e,a} D_{e,a} X A_{e,a}B_{e,a}.
\]
</div>

That extra term introduces several costs:

- Expert blocks are split into expert-adapter blocks.
- The number of active blocks grows from active experts to active expert-adapter
  pairs.
- Full-rank expert GEMMs are supplemented by skinny low-rank GEMMs.
- Each group reads its own \(A_g,B_g\), often for only a few tokens.
- The intermediate \(Z_g=X_gA_g\) may be written to and read from HBM if the
  path is not fused.
- Results must be scattered back and accumulated with router weights.

Top-k routing amplifies the effect. If each token routes to \(K\) experts and
LoRA is applied to \(P\) projections, then each token may trigger roughly:

<div class="math-display">
\[
K \times P \times 2
\]
</div>

low-rank fragments per MoE layer, where the factor 2 comes from the \(A\) and
\(B\) projections.

For \(K=2\) and \(P=3\), that is 12 low-rank fragments per token per MoE layer.

## Why It Is Slower Than Dense-LoRA

Dense-LoRA computes:

<div class="math-display">
\[
Y_{\Delta}^{\text{dense}} = XAB.
\]
</div>

All tokens share the same \(A,B\). That allows one globally batched shrink:

<div class="math-display">
\[
[N,d] \times [d,r],
\]
</div>

and one globally batched expand:

<div class="math-display">
\[
[N,r] \times [r,h].
\]
</div>

MoE-LoRA instead computes many group-specific projections:

<div class="math-display">
\[
\sum_{e,a}
[n_{e,a},d] \times [d,r],
\qquad
\sum_{e,a} n_{e,a} = N \times K.
\]
</div>

The total routed-token count may be comparable, but the batching is much worse.
Dense-LoRA has one adapter block. MoE-LoRA has many expert-adapter blocks.

Dense-LoRA also has no dynamic selector. MoE-LoRA needs \(D_{e,a}\), which means
token permutation, group construction, dynamic shapes, routing metadata, and
scatter-add. These are not part of the dense-LoRA path.

## Why Low Rank Does Not Automatically Save It

Low rank saves arithmetic. It does not automatically save runtime.

The factorization \(A_{e,a}B_{e,a}\) helps most when the basis is reused across
many rows of \(X\). Dense-LoRA has exactly that:

<div class="math-display">
\[
XAB.
\]
</div>

MoE-LoRA does not:

<div class="math-display">
\[
\sum_{e,a} D_{e,a} X A_{e,a}B_{e,a}.
\]
</div>

Each low-rank basis may serve only a small token group. The arithmetic is cheap,
but the operation becomes memory-bound, scheduling-bound, and fragmented.

That is why the core rule of thumb is:

<div class="math-display">
\[
\boxed{
\text{MoE-LoRA performance is largely controlled by the histogram of } n_{e,a}.
}
\]
</div>

Not just the mean. A batch with many tiny groups can be much slower than a batch
with fewer large groups, even if the total routed-token count is the same.

## The Structural Limit

Some of the slowdown is not an implementation bug.

In general, we cannot simplify:

<div class="math-display">
\[
\sum_{e,a}
D_{e,a} X A_{e,a}B_{e,a}
\]
</div>

into:

<div class="math-display">
\[
XAB,
\]
</div>

because \(D_{e,a}\) is token-dependent and the LoRA matrices vary by expert and
adapter.

Nor can we usually merge everything into one global delta matrix. For token
\(i\), the effective update is:

<div class="math-display">
\[
\Delta W_i^{\text{eff}}
=
\sum_{e \in \text{TopK}(i)}
\alpha_{i,e}
A_{e,a_i}B_{e,a_i}.
\]
</div>

The effective weight update changes from token to token. MoE-LoRA is therefore
an inherently row-wise dynamic operator.

The exact slowdown depends on kernels, scheduling, memory layout, and batching
policy. But the need for routing lookup, group construction, expert-adapter
weight lookup, low-rank block computation, and scatter-add is structural.

## What Good Kernels Try To Do

A naive implementation exposes the inefficient structure directly:

```python
Y_delta = zeros([N, h])

groups = group_by_expert_and_adapter(routes, adapter_ids)

for g in groups:
    token_ids = groups[g].token_ids
    router_weights = groups[g].router_weights

    Xg = X[token_ids]              # [n_g, d]

    Zg = Xg @ A[g]                 # [n_g, r]
    Yg = Zg @ B[g]                 # [n_g, h]

    Y_delta[token_ids] += router_weights[:, None] * Yg
```

This is mathematically clear but systems-hostile. It creates many small groups,
repeated reads of \(A_g,B_g\), an intermediate \(Z_g\), and scatter-add
pressure.

A better kernel tries to fuse the whole path:

```text
gather -> shrink -> expand -> router-weighted scatter-add
```

Ideally, it avoids materializing \(X_g\), \(Z_g\), or \(A_gB_g\) in global
memory. For decode-stage workloads, where \(n_g\) is often tiny, BGMV-style
batched gather matrix-vector kernels can be a better match than ordinary GEMM.

## Optimization Implications

The math points to a few practical directions:

- Increase \(n_{e,a}\): co-batch requests with the same adapter, limit adapter
  diversity per batch, and separate hot adapters from cold adapters.
- Reduce active blocks: apply LoRA only to selected experts or selected
  projections, prune cold expert-adapter pairs, or reduce top-k where quality
  allows.
- Fuse the LoRA path: keep \(Z_g=X_gA_g\) in registers or shared memory instead
  of writing it to HBM.
- Use small-group kernels: decode often looks more like batched gather
  matrix-vector multiplication than conventional GEMM.
- Improve weight layout: keep expert-adapter weights contiguous and aligned for
  the access pattern the runtime actually uses.
- Consider shared bases: replace independent \(A_{e,a}B_{e,a}\) with a shared
  structure such as \(UC_{e,a}V\), so parts of the low-rank computation can be
  reused.

The most important serving-level lever is often batching policy. If the runtime
mixes too many adapters in one decode batch, \(n_{e,a}\) collapses, and the
kernel has little reuse to exploit.

## Final Takeaway

MoE-LoRA is slower than base MoE because it adds:

<div class="math-display">
\[
\sum_{e,a}
D_{e,a} X A_{e,a}B_{e,a}
\]
</div>

on top of:

<div class="math-display">
\[
\sum_e D_e X W_e.
\]
</div>

It is slower than dense-LoRA because dense-LoRA computes \(XAB\) with one
globally shared low-rank basis, while MoE-LoRA computes many dynamic
expert-adapter-specific low-rank updates.

The key mathematical reason is fragmentation:

<div class="math-display">
\[
N
\rightarrow
n_e
\rightarrow
n_{e,a}.
\]
</div>

Dense-LoRA reuses \(A,B\) across \(N\) tokens. Base MoE reuses \(W_e\) across
\(n_e\) tokens. MoE-LoRA reuses \(A_{e,a},B_{e,a}\) only across \(n_{e,a}\)
tokens.

So the final summary is:

<div class="math-display">
\[
\boxed{
\text{MoE-LoRA is slow because low-rank structure and sparse routing interact badly.}
}
\]
</div>

Low rank reduces FLOPs. Sparse routing destroys batching and reuse. The result
is many small, skinny, dynamic matrix multiplications with low arithmetic
intensity.
