"""Memory-aware ONNX segment planning for the four multicard LLM exporters.

The KV alignment and context-linear scratch are adapted from
qwen3_5/estimate_plan_v2.py.  The internal high-water model is calibrated on
128-token RKNN reports and uses a peak of context and MoE workspaces, plus a
small shared baseline.  It is not a tensor-liveness analysis of RKNN kernels.
This module is intentionally free of torch/RKNN imports.
"""

from __future__ import annotations

import json
import os

MIB = 1024 * 1024
CTX_PAGE = 512
CARD_NOMINAL_MIB = {"rk1820": 2500, "rk1828": 5000}
CARD_USABLE_RATIO = 0.98
KV_COPIES = 8
INTERNAL_BYTES_PER_TOKEN = 6144
# The 32 MiB below is reserved for other card uses; it does not pay for
# model_report's command or internal allocations.
INTERNAL_BASE_BYTES = 16 * MIB
COMMAND_BYTES_PER_CARD = 16 * MIB
MOE_PREFILL_TOKENS = 128
FIXED_OVERHEAD_BYTES = 32 * MIB


def add_plan_arguments(parser):
    parser.add_argument(
        "--card", choices=sorted(CARD_NOMINAL_MIB), default="rk1828",
        help="target card for the memory estimate (default: rk1828)",
    )
    parser.add_argument(
        "--card_usable_mib", type=float, default=None,
        help="override usable memory per card in MiB (default: 98%% of nominal)",
    )


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _align_ctx(value):
    if value <= 0:
        return 0
    return _align_up(value, 32 if value < CTX_PAGE else CTX_PAGE)


def _floor_ctx(value):
    if value <= 0:
        return 0
    alignment = 32 if value < CTX_PAGE else CTX_PAGE
    return (value // alignment) * alignment


def _kv_slope_and_constant(head_dim, *, dtype="Int4_to_F16", payload_bits=4,
                           group_size=16, group_metadata_bytes=4,
                           residual_depth=64):
    # Match estimate_plan_v2: K uses MacC=32; V uses MacK=16 except F8.
    k_dim = _align_up(head_dim, 32)
    v_dim = _align_up(head_dim, 32 if dtype == "Int4_to_F8" else 16)
    total_dim = k_dim + v_dim
    slope = total_dim * payload_bits // 8
    if group_metadata_bytes:
        slope += total_dim // group_size * group_metadata_bytes
    constant = residual_depth * total_dim * 2 + 512 * 4
    return KV_COPIES * slope, KV_COPIES * constant


def _linear_state_bytes(config):
    nk = int(getattr(config, "linear_num_key_heads", 0) or 0)
    nv = int(getattr(config, "linear_num_value_heads", 0) or 0)
    kd = int(getattr(config, "linear_key_head_dim", 0) or 0)
    vd = int(getattr(config, "linear_value_head_dim", 0) or 0)
    conv = int(getattr(config, "linear_conv_kernel_dim", 0) or 0)
    return (2 * nk * kd + nv * vd) * conv * 2 + nv * kd * vd * 2 + 12288


def _qwen35_full_attention_params(config):
    """Full-layer parameter count used by estimate_plan_v2's prefill beta."""
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", None) or
                   hidden // config.num_attention_heads)
    q_dim = int(config.num_attention_heads) * head_dim
    kv_dim = int(getattr(config, "num_key_value_heads",
                         config.num_attention_heads)) * head_dim
    q_multiplier = 2 if getattr(config, "attn_output_gate", False) else 1
    attention = hidden * q_dim * q_multiplier + 2 * hidden * kv_dim + q_dim * hidden
    experts = int(getattr(config, "num_experts", 0) or 0)
    if experts:
        moe_inner = int(getattr(config, "moe_intermediate_size", 0) or 0)
        shared_inner = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        mlp = experts * 3 * hidden * moe_inner + hidden * experts
        mlp += 3 * hidden * shared_inner
    else:
        mlp = 3 * hidden * int(getattr(config, "intermediate_size", 0) or 0)
    return attention + mlp


def _layer_memory_terms(model_kind, config, num_layers):
    config = getattr(config, "text_config", None) or config
    head_dim = int(getattr(config, "head_dim", None) or
                   config.hidden_size // config.num_attention_heads)
    full_slope, full_constant = _kv_slope_and_constant(head_dim)
    if model_kind == "qwen3_5":
        layer_types = list(config.layer_types)
    elif model_kind == "gemma4":
        layer_types = list(config.layer_types)
    else:
        layer_types = ["full_attention"] * num_layers
    if len(layer_types) != num_layers:
        raise ValueError("config.layer_types must match the exported layer count")

    slopes, constants = [], []
    for layer_type in layer_types:
        if layer_type == "full_attention":
            slopes.append(full_slope)
            constants.append(full_constant)
        elif model_kind == "qwen3_5" and layer_type == "linear_attention":
            slopes.append(0)
            constants.append(_linear_state_bytes(config))
        elif model_kind == "gemma4" and layer_type == "sliding_attention":
            # The RKNN exporter always allocates a 1024-token Float16 window.
            sliding_slope, sliding_constant = _kv_slope_and_constant(
                head_dim, dtype="Float16", payload_bits=16,
                group_size=32, group_metadata_bytes=0, residual_depth=32,
            )
            slopes.append(0)
            constants.append(sliding_slope * 1024 + sliding_constant)
        else:
            raise ValueError(f"unsupported attention type: {layer_type}")
    return layer_types, slopes, constants


def _prefix(values):
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return result


def _moe_workspace_per_layer(model_kind, config, num_layers):
    """Empirical per-card MoE peak for the 128-token RKNN prefill graph.

    The expert gate/up FP16 logical size is only a scale proxy: the compiler
    may use additional routing/alignment buffers.  Qwen3/Qwen3.5's batched
    expert kernel and MiniCPM5's unpacked expert kernel need separate factors.
    The observed workspaces are reused between layers, so segment peaks use
    max, not sum.  This is an empirical peak approximation, not a claim that
    all individual buffers have disjoint lifetimes.
    """
    if model_kind not in ("qwen3", "qwen3_5", "minicpm5"):
        return [0] * num_layers
    experts = int(getattr(config, "num_experts", 0) or
                  getattr(config, "n_routed_experts", 0) or
                  getattr(config, "num_local_experts", 0) or 0)
    inner = int(getattr(config, "moe_intermediate_size", 0) or 0)
    if not experts or not inner:
        return [0] * num_layers
    gate_up_bytes = MOE_PREFILL_TOKENS * experts * (2 * inner) * 2
    # Conservative fits to the available model_report internal peaks:
    # batched Qwen3/3.5 experts: 1.6x; unpacked MiniCPM5 experts: 5.1x.
    numerator = 51 if model_kind == "minicpm5" else 16
    peak = (gate_up_bytes * numerator + 9) // 10
    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    return [0 if index < first_dense else peak for index in range(num_layers)]


def _internal_estimate(context, moe_workspace):
    return INTERNAL_BASE_BYTES + max(INTERNAL_BYTES_PER_TOKEN * context,
                                     moe_workspace)


def _segment_context_limit(capacity, base, kv_slope, moe_workspace):
    """Solve base + baseline + kv*ctx + max(ctx_scratch, MoE) <= capacity."""
    room = capacity - base - INTERNAL_BASE_BYTES
    if room < moe_workspace:
        return -1
    linear_limit = room // (kv_slope + INTERNAL_BYTES_PER_TOKEN)
    if kv_slope:
        moe_limit = (room - moe_workspace) // kv_slope
        return min(linear_limit, moe_limit)
    return linear_limit


def _max_context_partition(weights, slopes, constants, last_extra, num_segments,
                           capacity_bytes, moe_workspaces=None):
    """Maximize the minimum per-segment context limit over contiguous cuts."""
    n = len(weights)
    if moe_workspaces is None:
        moe_workspaces = [0] * n
    wp, sp, cp = map(_prefix, (weights, slopes, constants))
    dp = {0: float("inf")}
    back = [{}]
    for k in range(1, num_segments + 1):
        cur, parents = {}, {}
        for end in range(k, n - (num_segments - k) + 1):
            for start in range(k - 1, end):
                if start not in dp:
                    continue
                base = (wp[end] - wp[start] + cp[end] - cp[start]
                        + FIXED_OVERHEAD_BYTES + COMMAND_BYTES_PER_CARD
                        + (last_extra if k == num_segments else 0))
                limit = _segment_context_limit(
                    capacity_bytes, base, sp[end] - sp[start],
                    max(moe_workspaces[start:end]))
                candidate = min(dp[start], limit)
                if end not in cur or candidate > cur[end]:
                    cur[end], parents[end] = candidate, start
        dp = cur
        back.append(parents)
    if n not in dp:
        return 0, None
    counts, end = [], n
    for k in range(num_segments, 0, -1):
        start = back[k][end]
        counts.append(end - start)
        end = start
    counts.reverse()
    return _floor_ctx(int(dp[n])), counts


def _fixed_split_max_context(counts, weights, slopes, constants, last_extra,
                             capacity_bytes, moe_workspaces=None):
    start, limit = 0, None
    if moe_workspaces is None:
        moe_workspaces = [0] * len(weights)
    for index, count in enumerate(counts):
        end = start + count
        base = (sum(weights[start:end]) + sum(constants[start:end])
                + FIXED_OVERHEAD_BYTES + COMMAND_BYTES_PER_CARD
                + (last_extra if index == len(counts) - 1 else 0))
        card_limit = _segment_context_limit(
            capacity_bytes, base, sum(slopes[start:end]),
            max(moe_workspaces[start:end]))
        limit = card_limit if limit is None else min(limit, card_limit)
        start = end
    return _floor_ctx(limit or 0)


def _balanced_counts(objective, memory, last_obj, last_mem, num_segments,
                     memory_budget):
    """Two-stage minimax deviation DP from estimate_plan_v2, with a hard cap."""
    n = len(objective)
    op, mp = _prefix(objective), _prefix(memory)
    total = op[-1] + last_obj

    def deviation(start, end, k):
        return abs((op[end] - op[start]
                    + (last_obj if k == num_segments else 0)) * num_segments
                   - total)

    def feasible(start, end, k):
        return (mp[end] - mp[start]
                + (last_mem if k == num_segments else 0)) <= memory_budget

    previous = {0: 0}
    for k in range(1, num_segments + 1):
        current = {}
        for end in range(k, n - (num_segments - k) + 1):
            for start, prior in previous.items():
                if start >= end or not feasible(start, end, k):
                    continue
                candidate = max(prior, deviation(start, end, k))
                if end not in current or candidate < current[end]:
                    current[end] = candidate
        previous = current
    if n not in previous:
        return None
    best_deviation = previous[n]
    previous = {0: (0, [])}
    for k in range(1, num_segments + 1):
        current = {}
        for end in range(k, n - (num_segments - k) + 1):
            for start, (prior, counts) in previous.items():
                if start >= end or not feasible(start, end, k):
                    continue
                dev = deviation(start, end, k)
                if dev > best_deviation:
                    continue
                candidate = prior + dev * dev
                if end not in current or candidate < current[end][0]:
                    current[end] = (candidate, counts + [end - start])
        previous = current
    return previous[n][1] if n in previous else None


def _validate_manual_counts(counts, num_layers):
    try:
        result = [int(value) for value in counts]
    except (TypeError, ValueError) as exc:
        raise ValueError("--segment_layer_counts must contain integers") from exc
    if not result or any(value <= 0 for value in result) or sum(result) != num_layers:
        raise ValueError("--segment_layer_counts must contain positive integers "
                         f"summing to {num_layers}")
    return result


def _mib(value):
    return round(value / MIB, 2)


def make_plan(model_kind, config, layer_weight_bytes, last_weight_bytes, args):
    """Return a plan whose split is directly usable as --segment_layer_counts."""
    weights = [int(value) for value in layer_weight_bytes]
    n = len(weights)
    if n == 0 or any(value < 0 for value in weights):
        raise ValueError("a non-empty list of nonnegative layer weights is required")
    last_extra = int(last_weight_bytes)
    manual = getattr(args, "segment_layer_counts", None)
    if manual is not None:
        counts = _validate_manual_counts(manual, n)
        num_segments = len(counts)
        mode = "manual"
    else:
        num_segments = int(getattr(args, "num_segments", 0) or 0)
        if num_segments <= 0:
            num_segments = (n + 7) // 8
        num_segments = min(num_segments, n)
        mode = "automatic"
    card = getattr(args, "card", "rk1828")
    if card not in CARD_NOMINAL_MIB:
        raise ValueError(f"unsupported card: {card}")
    usable_mib = getattr(args, "card_usable_mib", None)
    if usable_mib is None:
        usable_mib = CARD_NOMINAL_MIB[card] * CARD_USABLE_RATIO
    if not 0 < usable_mib <= CARD_NOMINAL_MIB[card]:
        raise ValueError("--card_usable_mib must be positive and at most the "
                         "card's nominal capacity")
    capacity = int(usable_mib * MIB)
    text_config = getattr(config, "text_config", None) or config
    layer_types, slopes, constants = _layer_memory_terms(model_kind, text_config, n)
    moe_workspaces = _moe_workspace_per_layer(model_kind, text_config, n)
    model_limit = int(getattr(text_config, "max_position_embeddings", 0) or 0)

    if mode == "automatic":
        unconstrained_ctx, optimal_counts = _max_context_partition(
            weights, slopes, constants, last_extra, num_segments, capacity,
            moe_workspaces)
        context = (min(unconstrained_ctx, model_limit)
                   if model_limit else unconstrained_ctx)
        if context > 0:
            aligned = _align_ctx(context)
            kv = [s * aligned + c for s, c in zip(slopes, constants)]
            memory = [w + k for w, k in zip(weights, kv)]
            # Prefill objective: attention work scales with ctx * heads *
            # head_dim. Qwen3.5 uses the source estimator's parameter formula;
            # other models use the W4/group32 byte-per-parameter approximation.
            heads = int(getattr(text_config, "num_attention_heads", 0) or 0)
            head_dim = int(getattr(text_config, "head_dim", None) or
                           text_config.hidden_size // heads)
            full_params = (_qwen35_full_attention_params(text_config)
                           if model_kind == "qwen3_5" else 0)
            objective = []
            for w, k, layer_type in zip(weights, kv, layer_types):
                attention_bonus = 0
                if layer_type == "full_attention":
                    if full_params:
                        # Exact beta from estimate_plan_v2 for Qwen3.5.
                        attention_bonus = round(
                            w * (2 * context * heads * head_dim / full_params))
                    else:
                        # Other models: approximate W4/group32 bytes/parameter.
                        attention_bonus = 2 * context * heads * head_dim * 19 // 32
                objective.append(w + k + attention_bonus)
            budget = (capacity - FIXED_OVERHEAD_BYTES - COMMAND_BYTES_PER_CARD
                      - INTERNAL_BASE_BYTES
                      - max(INTERNAL_BYTES_PER_TOKEN * aligned,
                            max(moe_workspaces)))
            norm_objective = min(last_extra, 2 * int(text_config.hidden_size))
            last_objective = norm_objective + (last_extra - norm_objective) // 128
            counts = _balanced_counts(objective, memory,
                                      last_objective, last_extra,
                                      num_segments, budget)
            if counts is None:
                counts = optimal_counts
        else:
            counts = optimal_counts or _balanced_counts(
                weights, weights, last_extra, last_extra, num_segments,
                sum(weights) + last_extra)
        assert counts is not None
    fixed_max = _fixed_split_max_context(
        counts, weights, slopes, constants, last_extra, capacity, moe_workspaces)
    context = min(fixed_max, model_limit) if model_limit else fixed_max
    aligned = _align_ctx(context)
    scratch = INTERNAL_BYTES_PER_TOKEN * aligned
    cards, start = [], 0
    for index, count in enumerate(counts):
        end = start + count
        block_weights = sum(weights[start:end])
        head_weights = last_extra if index == num_segments - 1 else 0
        kv_bytes = sum(s * aligned + c for s, c in
                       zip(slopes[start:end], constants[start:end]))
        moe_workspace = max(moe_workspaces[start:end])
        internal = _internal_estimate(aligned, moe_workspace)
        total = (block_weights + head_weights + kv_bytes + internal
                 + COMMAND_BYTES_PER_CARD + FIXED_OVERHEAD_BYTES)
        cards.append({
            "card_index": index, "layer_start": start, "layer_end_exclusive": end,
            "layer_count": count,
            "full_attention_layers": layer_types[start:end].count("full_attention"),
            "sliding_attention_layers": layer_types[start:end].count("sliding_attention"),
            "linear_attention_layers": layer_types[start:end].count("linear_attention"),
            "block_weights_mib": _mib(block_weights),
            "norm_lm_head_weights_mib": _mib(head_weights),
            "kv_cache_mib": _mib(kv_bytes),
            "internal_scratch_mib": _mib(scratch),
            "internal_moe_workspace_mib": _mib(moe_workspace),
            "internal_base_mib": _mib(INTERNAL_BASE_BYTES),
            "internal_estimate_mib": _mib(internal),
            "command_estimate_mib": _mib(COMMAND_BYTES_PER_CARD),
            "fixed_overhead_mib": _mib(FIXED_OVERHEAD_BYTES),
            "total_mib": _mib(total),
            "remaining_mib": _mib(capacity - total),
        })
        start = end
    stop = getattr(args, "stop_after_segment", None)
    if stop is not None and not 0 <= stop < num_segments:
        raise ValueError("--stop_after_segment must be within the planned segment range")
    return {
        "model": model_kind, "plan_mode": mode, "card": card,
        "card_nominal_mib": CARD_NOMINAL_MIB[card],
        "card_usable_mib": round(usable_mib, 2),
        "num_segments": num_segments, "segment_layer_counts": counts,
        "max_context_len": context,
        "suggested_MAX_CONTEXT_LEN": context,
        "max_context_len_uncapped": fixed_max,
        "model_max_position_embeddings": model_limit or None,
        "context_alignment": 32 if context < CTX_PAGE else CTX_PAGE,
        "feasible": context > 0,
        "per_card": cards,
        "exported_segment_indices": list(range(stop + 1 if stop is not None else num_segments)),
        "assumptions": {
            "weight_quantization": "W4A16/group32 blocks, W6A16/group32 lm_head, FP16 norm",
            "kv_copies_per_attention_layer": KV_COPIES,
            "internal_scratch_bytes_per_token_per_card": INTERNAL_BYTES_PER_TOKEN,
            "internal_base_mib_per_card": _mib(INTERNAL_BASE_BYTES),
            "moe_workspace_prefill_tokens": MOE_PREFILL_TOKENS,
            "moe_workspace_factor_batched": 1.6,
            "moe_workspace_factor_minicpm5_unpacked": 5.1,
            "command_estimate_mib_per_card": _mib(COMMAND_BYTES_PER_CARD),
            "fixed_overhead_mib_per_card": _mib(FIXED_OVERHEAD_BYTES),
            "fixed_overhead_purpose": "system reserve; separate from internal and command",
            "calibration": "RKNN 128-token reports: Qwen3 30B MoE, Qwen3.5 35B MoE at 4K and 100K, MiniCPM5 15B MoE",
            "warning": "Reference only; confirm with each card's RKNN model_report before deployment.",
        },
    }


def prepare_plan(model_kind, config, layer_weight_bytes, last_weight_bytes, args):
    plan = make_plan(model_kind, config, layer_weight_bytes, last_weight_bytes, args)
    args.segment_layer_counts = plan["segment_layer_counts"]
    args._segment_plan = plan
    print(f"Planned layer distribution: {plan['segment_layer_counts']}")
    print(f"Estimated usable {plan['card']}: {plan['card_usable_mib']} MiB/card; "
          f"MAX_CONTEXT_LEN <= {plan['max_context_len']} (reference only)")
    if not plan["feasible"]:
        print("WARNING: estimated split does not fit even a small RKNN context; "
              "increase card count or usable capacity.")
    return plan


def write_plan_json(args):
    plan = getattr(args, "_segment_plan", None)
    if plan is None:
        return None
    path = os.path.splitext(args.export_llm_path)[0] + ".segment_plan.json"
    with open(path, "w", encoding="utf-8") as output:
        json.dump(plan, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Segment memory/context estimate saved to {os.path.abspath(path)}")
    return path
