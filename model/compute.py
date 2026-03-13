"""Analytical compute-efficiency utilities for model comparison.

Pure functions (no JAX dependency) that compute parameter counts, FLOPs,
and memory estimates from config parameters.
"""


def transformer_param_count(
    *,
    n_vocab,
    n_layers,
    n_hidden,
    n_heads,
    use_swiglu,
    use_bias,
    n_out,
    n_pred_tokens,
    n_mlp_hidden=None,
):
    """Compute transformer parameter count analytically.

    Returns dict with keys: total, embedding, per_block, blocks_total,
    output_head, final_norm.
    """
    if n_mlp_hidden is None:
        if use_swiglu:
            n_mlp_hidden = int(4 * n_hidden * 2 / 3)
        else:
            n_mlp_hidden = 4 * n_hidden

    # Embedding: n_vocab * n_hidden
    embedding = n_vocab * n_hidden

    # Per block attention: combined QKV + out_proj
    attn = (n_hidden * 3 * n_hidden + n_hidden * n_hidden)
    if use_bias:
        attn += 3 * n_hidden + n_hidden

    # Per block MLP
    if use_swiglu:
        # gate(D->M), up(D->M), down(M->D)
        mlp = 3 * n_hidden * n_mlp_hidden
        if use_bias:
            mlp += 2 * n_mlp_hidden + n_hidden
    else:
        # fc1(D->M), fc2(M->D)
        mlp = 2 * n_hidden * n_mlp_hidden
        if use_bias:
            mlp += n_mlp_hidden + n_hidden

    # Two RMSNorms per block (scale only)
    norms = 2 * n_hidden

    per_block = attn + mlp + norms

    # Final RMSNorm
    final_norm = n_hidden

    # Output head
    out_features = n_out * n_pred_tokens
    output_head = n_hidden * out_features
    if use_bias:
        output_head += out_features

    blocks_total = n_layers * per_block
    total = embedding + blocks_total + final_norm + output_head

    return {
        "total": total,
        "embedding": embedding,
        "per_block": per_block,
        "blocks_total": blocks_total,
        "output_head": output_head,
        "final_norm": final_norm,
    }


def mamba2_bonsai_param_count(
    *,
    n_vocab,
    n_layers,
    n_hidden,
    n_heads,
    d_state,
    d_conv,
    expand,
    use_bias,
    n_out,
    n_pred_tokens,
):
    """Compute Mamba2 Bonsai parameter count analytically.

    Derived dims (ref ssm_bonsai.py):
      E = expand * n_hidden
      P = E // n_heads
      proj_size = 2*(E + d_state) + n_heads  (line 302)
      conv_dim = E + 2*d_state               (line 312)
    """
    E = expand * n_hidden
    proj_size = 2 * (E + d_state) + n_heads
    conv_dim = E + 2 * d_state

    # Embedding
    embedding = n_vocab * n_hidden

    # Per block
    # in_proj: D -> proj_size
    in_proj = n_hidden * proj_size
    if use_bias:
        in_proj += proj_size

    # depthwise conv1d: kernel (d_conv, 1, conv_dim) = d_conv * conv_dim
    conv1d = d_conv * conv_dim
    if use_bias:
        conv1d += conv_dim

    # RMSNormGate: weight only, size E
    norm_gate = E

    # out_proj: E -> D
    out_proj = E * n_hidden
    if use_bias:
        out_proj += n_hidden

    # SSM scalars: a_log, d, dt_bias (each n_heads)
    ssm_scalars = 3 * n_heads

    # pre_norm (RMSNorm): scale only, size D
    pre_norm = n_hidden

    per_block = in_proj + conv1d + norm_gate + out_proj + ssm_scalars + pre_norm

    # Final RMSNorm
    final_norm = n_hidden

    # Output head
    out_features = n_out * n_pred_tokens
    output_head = n_hidden * out_features
    if use_bias:
        output_head += out_features

    blocks_total = n_layers * per_block
    total = embedding + blocks_total + final_norm + output_head

    return {
        "total": total,
        "embedding": embedding,
        "per_block": per_block,
        "blocks_total": blocks_total,
        "output_head": output_head,
        "final_norm": final_norm,
    }


# --- Forward FLOPs (single forward pass, batch_size=1) ---


def transformer_flops_forward(
    *,
    n_seq,
    n_layers,
    n_hidden,
    n_heads,
    use_swiglu,
    n_vocab,
    n_mlp_hidden=None,
):
    """Compute transformer forward FLOPs analytically.

    Per layer attention: 8*S*D² + 4*S²*D
      (QKV proj 6SD², Q@K^T 2S²D, attn@V 2S²D, out_proj 2SD²)
    Per layer MLP (SwiGLU): 6*S*D*M  where M = int(4*D*2/3)
    Per layer MLP (GELU):   4*S*D*M  where M = 4*D
    Output head: 2*S*D*V

    Quadratic term: 4*S²*D per layer from attention.
    """
    S, D, V = n_seq, n_hidden, n_vocab

    if n_mlp_hidden is None:
        M = int(4 * D * 2 / 3) if use_swiglu else 4 * D
    else:
        M = n_mlp_hidden

    attn_per_layer = 8 * S * D * D + 4 * S * S * D
    mlp_per_layer = (6 if use_swiglu else 4) * S * D * M
    per_layer = attn_per_layer + mlp_per_layer
    all_layers = n_layers * per_layer
    output_head = 2 * S * D * V
    total = all_layers + output_head

    return {
        "total": total,
        "attn_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "per_layer": per_layer,
        "all_layers": all_layers,
        "output_head": output_head,
    }


def mamba2_bonsai_flops_forward(
    *,
    n_seq,
    n_layers,
    n_hidden,
    n_heads,
    d_state,
    d_conv,
    expand,
    scan_chunk_len,
    n_vocab,
):
    """Compute Mamba2 Bonsai forward FLOPs analytically.

    Per layer projections (linear in S):
      in_proj:  2*S*D*proj_size
      conv1d:   2*S*d_conv*conv_dim
      out_proj: 2*S*E*D

    SSD scan einsums (ref ssm_bonsai.py):
      Intra-chunk  (line 158): S*C*H*(N+P)
      State accum  (line 161): S*H*P*N
      Inter-chunk  (line 170): H*(S/C)²*P*N   (small quadratic)
      State-to-out (line 174): S*H*N*P

    Dominant term is linear in S; quadratic coefficient ~170x smaller
    than transformer for typical configs.
    """
    S, D, V = n_seq, n_hidden, n_vocab
    E = expand * D
    P = E // n_heads
    N = d_state
    H = n_heads
    C = scan_chunk_len

    proj_size = 2 * (E + N) + H
    conv_dim = E + 2 * N

    # Per layer projections
    in_proj_flops = 2 * S * D * proj_size
    conv_flops = 2 * S * d_conv * conv_dim
    out_proj_flops = 2 * S * E * D
    proj_per_layer = in_proj_flops + conv_flops + out_proj_flops

    # SSD scan
    n_chunks = max(1, -(-S // C))  # ceil division
    ssd_intra = S * C * H * (N + P)
    ssd_state_accum = S * H * P * N
    ssd_inter = H * n_chunks * n_chunks * P * N
    ssd_state_out = S * H * N * P
    ssd_per_layer = ssd_intra + ssd_state_accum + ssd_inter + ssd_state_out

    per_layer = proj_per_layer + ssd_per_layer
    all_layers = n_layers * per_layer
    output_head = 2 * S * D * V
    total = all_layers + output_head

    return {
        "total": total,
        "proj_per_layer": proj_per_layer,
        "ssd_per_layer": ssd_per_layer,
        "per_layer": per_layer,
        "all_layers": all_layers,
        "output_head": output_head,
        "ssd_intra": ssd_intra,
        "ssd_inter": ssd_inter,
    }


# --- Training & memory helpers ---


def training_flops_total(forward_flops, *, train_iters, batch_size, grad_accum_steps=1):
    """Total training FLOPs.  Backward ≈ 2× forward, so total ≈ 3× forward."""
    return 3 * forward_flops * batch_size * grad_accum_steps * train_iters


def memory_bytes_estimate(
    n_params,
    *,
    batch_size,
    n_seq,
    n_hidden,
    n_layers,
    n_heads=None,
    model_family="transformer",
    param_dtype_bytes=4,
    compute_dtype_bytes=2,
):
    """Estimate peak training memory in bytes.

    Components:
      - Model params:  n_params * param_dtype_bytes
      - AdamW state:   2 * n_params * 4  (m and v always float32)
      - Activations:   per-layer residual stream + attention/SSM state
    """
    params_bytes = n_params * param_dtype_bytes
    optimizer_bytes = 2 * n_params * 4

    residual_per_layer = batch_size * n_seq * n_hidden * compute_dtype_bytes
    if model_family == "transformer" and n_heads is not None:
        attn_per_layer = batch_size * n_heads * n_seq * n_seq * compute_dtype_bytes
    else:
        attn_per_layer = 0

    activations_bytes = n_layers * (residual_per_layer + attn_per_layer)
    total_bytes = params_bytes + optimizer_bytes + activations_bytes

    return {
        "params_bytes": params_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activations_bytes": activations_bytes,
        "total_bytes": total_bytes,
    }


def inference_activation_bytes(
    *,
    n_seq,
    n_hidden,
    n_layers,
    n_heads,
    model_family="transformer",
    d_state=None,
    d_conv=None,
    expand=2,
    compute_dtype_bytes=2,
):
    """Peak activation memory (bytes) for prefill + single next-token prediction.

    Transformer: O(S²) — attention matrix materialization during prefill.
    Mamba2: O(1) w.r.t. S — SSM recurrent state + conv state (independent of S).
    """
    if model_family == "transformer":
        residual_per_layer = n_seq * n_hidden * compute_dtype_bytes
        attn_per_layer = n_heads * n_seq * n_seq * compute_dtype_bytes
        return n_layers * (residual_per_layer + attn_per_layer)
    elif model_family == "mamba2_bonsai":
        head_dim = (expand * n_hidden) // n_heads
        ssm_per_layer = n_heads * head_dim * d_state * compute_dtype_bytes
        conv_dim = expand * n_hidden + 2 * d_state
        conv_per_layer = conv_dim * (d_conv - 1) * compute_dtype_bytes
        return n_layers * (ssm_per_layer + conv_per_layer)
    else:
        raise ValueError(f"Unknown model_family: {model_family!r}")


# --- Unified dispatcher ---


def compute_metrics_from_info(info, *, n_seq_override=None):
    """Compute analytical metrics from an experiment info dict.

    Reads model_family from info and dispatches to the appropriate
    param-count and FLOPs functions.

    Returns flat dict: {n_params, forward_flops, params, flops}.
    """
    model_family = info.get("model_family")
    n_vocab = int(info.get("n_vocab", 128))
    n_layers = int(info.get("n_layers", 2))
    n_hidden = int(info.get("n_hidden", 128))
    n_heads = int(info.get("n_heads", 4))
    use_bias = bool(info.get("use_bias", True))
    n_out = int(info.get("n_out", n_vocab))
    n_pred_tokens = int(info.get("n_pred_tokens", 1))
    n_seq = int(n_seq_override if n_seq_override is not None else info.get("n_seq", 128))

    if model_family == "transformer":
        use_swiglu = bool(info.get("use_swiglu", False))
        params = transformer_param_count(
            n_vocab=n_vocab, n_layers=n_layers, n_hidden=n_hidden,
            n_heads=n_heads, use_swiglu=use_swiglu, use_bias=use_bias,
            n_out=n_out, n_pred_tokens=n_pred_tokens,
        )
        flops = transformer_flops_forward(
            n_seq=n_seq, n_layers=n_layers, n_hidden=n_hidden,
            n_heads=n_heads, use_swiglu=use_swiglu, n_vocab=n_vocab,
        )
    elif model_family == "mamba2_bonsai":
        d_state = int(info.get("d_state", 16))
        d_conv = int(info.get("d_conv", 4))
        expand = int(info.get("expand", 2))
        scan_chunk_len = int(info.get("scan_chunk_len", 64))
        params = mamba2_bonsai_param_count(
            n_vocab=n_vocab, n_layers=n_layers, n_hidden=n_hidden,
            n_heads=n_heads, d_state=d_state, d_conv=d_conv,
            expand=expand, use_bias=use_bias,
            n_out=n_out, n_pred_tokens=n_pred_tokens,
        )
        flops = mamba2_bonsai_flops_forward(
            n_seq=n_seq, n_layers=n_layers, n_hidden=n_hidden,
            n_heads=n_heads, d_state=d_state, d_conv=d_conv,
            expand=expand, scan_chunk_len=scan_chunk_len, n_vocab=n_vocab,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family!r}")

    return {
        "n_params": params["total"],
        "forward_flops": flops["total"],
        "params": params,
        "flops": flops,
    }
