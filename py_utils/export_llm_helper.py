import os
import torch
import numpy as np

import onnxruntime as ort

def register_bitwise_right_shift():
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper

    # 定义自定义的 symbolic 函数
    def custom_rshift(g, self, other):
        self_type = symbolic_helper._try_get_scalar_type(self).onnx_type()

        # 确保 `other` 转换为 `self` 的数据类型
        if symbolic_helper._try_get_scalar_type(other) != symbolic_helper._try_get_scalar_type(self):
            other = g.op("Cast", other, to_i=self_type)

        # 处理 uint8 类型
        if symbolic_helper._try_get_scalar_type(self) == torch.uint8:
            return g.op("BitShift", self, other, direction_s="RIGHT")

        # 其他类型按位右移逻辑
        two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
        if not symbolic_helper._is_fp(self):
            other = g.op("Cast", other, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        two_pow = g.op("Pow", two, other)
        two_pow = g.op("Cast", two_pow, to_i=self_type)
        # rshift = g.op("Div", self, two_pow)   # Div是向零取整, 但bitwise_right_shift是向下取整, 因此采用下面的方式实现向下取整

        # 向下取整实现
        div_result = g.op("Div", self, two_pow)
        sign_a = g.op("Min", g.op("Sign", self), g.op("Constant", value_t=torch.tensor(0)))  # 转换为 0 或 -1
        remainder = g.op("Mod", g.op("Abs", self), two_pow)
        has_remainder = g.op("Cast", g.op("Greater", remainder, g.op("Constant", value_t=torch.tensor(0))), to_i=self_type)
        rshift = g.op("Add", div_result, g.op("Mul", sign_a, has_remainder))

        return rshift

    # 注册自定义的 symbolic 函数
    register_custom_op_symbolic("aten::bitwise_right_shift", custom_rshift, 11)

def check_gptq(bit, group_size):
    if bit == 4 and group_size in [-1, 32, 64, 128]:
        return True
    return False


def causal_qwen_3_5_llm_to_onnx(model, args):
    import torch
    import os

    messages = args.chat_context
    tokenizer = args.tokenizer
    out_path = args.export_llm_path
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 封装在提示词模板里，所有模型默认，function calling功能也依赖提示词模板
        enable_thinking=True,  # 是否启动思考模式，也是通过修改提示词模板实现
    )

    # 1. 构造模型推理所需的张量输入
    # 注意：这里的 model 是包裹后的 Qwen3_5_export_model_cst，我们要用它内部的原始模型(model.model)做初始推理
    device = model.model.device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    src_len = model_inputs['input_ids'].shape[1]
    
    input_ids = model_inputs['input_ids']
    full_attention_mask = model_inputs['attention_mask'].to(torch.float)
    position_ids = torch.arange(src_len, dtype=torch.long, device=device).unsqueeze(0)
    num_logits_to_keep = torch.tensor([src_len-1], dtype=torch.long, device=device)
    tc_in = torch.tensor([src_len], dtype=torch.int64, device=device)

    # 2. 跑一次原始模型的前向推理，以此获取合法的 past_key_values / conv_states
    with torch.no_grad():
        model_out = model.model(
            input_ids=input_ids, 
            attention_mask=full_attention_mask, 
            position_ids=position_ids, 
            num_logits_to_keep=num_logits_to_keep, 
            Tc=tc_in
        )

    # 3. 提取用于导出的状态变量
    past_key_values = model_out.past_key_values
    conv_status = torch.concat([v for v in past_key_values.conv_states if v is not None], dim=0)

    # 4. 配置 ONNX 输入输出名称及动态轴
    input_names = ['input_ids', 'position_ids', 'attention_mask', 'Linear_Tc', 'num_logits_to_keep', 'conv_state']
    basic_in_len = 5 # 在 conv_state 之前的输入个数
    
    output_names = ['output'] + [n+'_out' for n in input_names[basic_in_len:]]
    dynamic_axes = {
        'attention_mask': {1: 'seq_len'}, 
        'position_ids': {1: 'seq_len'}, 
        'input_ids': {1: 'seq_len'}
    }

    print(f"Exporting model to {out_path} ...")
    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if check_gptq(q_config.bits, q_config.group_size) == False:
            print("GRQ model quantization not supported. Only W4A16 quantization grouped or channel asymmetric/symmetric with group_size in {32, 64, 128} or -1 (per-channel) is supported.")
            exit(1)
        register_bitwise_right_shift()

    # 5. 导出 ONNX（整合并清理了之前重复的代码块）
    with torch.no_grad():
        torch.onnx.export(
            model, # 传入包裹后的导出专用模型
            (
                input_ids,
                position_ids,
                full_attention_mask,
                tc_in,
                num_logits_to_keep,
                conv_status,
            ),
            out_path,
            export_params=True,             # 做参数导出
            do_constant_folding=True,       # 做常量折叠优化
            opset_version=19,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            custom_opsets={"rknn": 1},      # 声明自定义域 (适配 NPU 等场景)
        )
    print("Export completed successfully!")


def causal_llm_to_onnx(model, args):
    import torch

    # Debug Parameter
    args.prompt_size = 64
    args.dynamic_shape = True

    model.eval()
    in_len = args.prompt_size

    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask = torch.ones((1, in_len), dtype=torch.float)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
        dummy_input = torch.zeros((1, in_len, args.hidden_size), dtype=torch.float)
        position_ids = torch.zeros((1, 1, in_len), dtype=torch.long)

    inputs = (dummy_input, attention_mask, position_ids)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
        input_names = ["input_embeds", "attention_mask", "position_ids"]
    dynamic_axes = {}
    if args.dynamic_shape:
        if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
            dynamic_axes.update({
                'input_embeds': {1: 'sequence'},
                'attention_mask': {1: 'sequence'},
                'position_ids': {2: 'sequence'},
            })
        else:
            dynamic_axes.update({
                'input_ids': {1: 'sequence'},
                'attention_mask': {1: 'sequence'},
                'position_ids': {1: 'sequence'},
            })

    # 获取 forward 参数
    forward_args = model.forward.__code__.co_varnames
    for i in range(3):
        name = f"deepstack_embeds{i}"
        if name in forward_args:
            embed = torch.randn(1, in_len, args.hidden_size, dtype=torch.float32)
            inputs = (*inputs, embed)
            input_names.append(name)
            dynamic_axes[name] = {1: "sequence"}

    output_names = ["output"]

    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key  = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__

    for key in logit_keep_keys:
        if key in _forward_func.__code__.co_varnames:
            logit_keep_key = key
            break
    if logit_keep_key:
        # 只留最后一个 token 的 logits，减少计算量
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        insert_nones = [None]* (_forward_func.__code__.co_varnames.index(logit_keep_key) - len(inputs) -1)
        inputs = (*inputs, *insert_nones, num_logits_to_keep)
        input_names.append('num_logits_to_keep')


    if getattr(args, 'output_hidden_states', False) and 'output_hidden_states' in _forward_func.__code__.co_varnames:
        idx = _forward_func.__code__.co_varnames.index('output_hidden_states') - 1
        if idx < len(inputs):
            # 如果有 output_hidden_states 参数，则需要在输入中添加一个 None
            inputs = (*inputs[:idx], True, *inputs[idx+1:])
        else:
            inputs = (*inputs, *((None,)*(_forward_func.__code__.co_varnames.index('output_hidden_states') - len(inputs) - 1)), True)

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if check_gptq(q_config.bits, q_config.group_size) == False:
            print("GRQ model quantization not supported. Only W4A16 quantization grouped or channel asymmetric/symmetric with group_size in {32, 64, 128} or -1 (per-channel) is supported.")
            exit(1)
        register_bitwise_right_shift()
    else:
        model.float()

    # out = model(*inputs)
    # if len(out) != len(output_names):
    #     print(f"WARNING: output number not match, expect {len(output_names)}, got {len(out)}")
    #     print(f"WARNING: try only keep one output")
    #     output_names = output_names[:1]

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    if False:
        ort.set_default_logger_severity(3)  # 设置 ONNX Runtime 日志级别为 WARNING 及以上
        # class MyLoggingHandler(ort.LoggingHandler):
        #     def __init__(self):
        #         super().__init__()
            
        #     def log(self, severity, category, logid, code_location, message):
        #         # 在这里自定义处理日志消息
        #         if severity < 3:  # 只记录 WARNING 及以上级别的日志
        #             print(f"[ONNX Runtime] {message}")

        # # 注册自定义日志处理器
        # ort.set_default_logger(MyLoggingHandler())

        # check onnx model result
        print("Checking ONNX model output...")
        sess_onnx = ort.InferenceSession(args.export_llm_path, providers=['CPUExecutionProvider'])
        valid_inputs = [v.numpy() if isinstance(v, torch.Tensor) else v for v in inputs if v is not None]
        input_feed   = {sess_onnx.get_inputs()[i].name: valid_inputs[i] for i in range(len(valid_inputs))}
        for k, v in input_feed.items():
            if isinstance(v, np.ndarray):
                print(f"Input {k}: {v.shape} {v.dtype} {v}")
            else:
                print(f"Input {k}: {type(v)} {v}")

        output_onnx  = sess_onnx.run(output_names, input_feed)
        output_name  = sess_onnx.get_outputs()[0].name
        cos_sim      = torch.cosine_similarity(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), dim=-1)
        eluer_dist   = torch.dist(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), p=2)
        abs_diff     = torch.abs(torch.tensor(output_onnx[0]) - out[0].detach().cpu())
        print(f"Cosine Similarity       : {cos_sim.item()}")
        print(f"Euclidean Distance      : {eluer_dist.item()}")
        print(f"Max Absolute Difference : {abs_diff.max().item()}")
        print("first 10 elements of output:")
        print(f"ONNX output: {output_onnx[0].reshape(-1)[:10]}")
        print(f"PyTorch output: {out[0].detach().cpu().numpy().reshape(-1)[:10]}")

        if True:
            save_dir = os.path.join(os.path.dirname(args.export_llm_path), "src_io")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for k, v in input_feed.items():
                if isinstance(v, np.ndarray):
                    np.save(os.path.join(save_dir, f"{k}.npy"), v)
                else:
                    with open(os.path.join(save_dir, f"{k}.txt"), 'w') as f:
                        f.write(str(v))
            np.save(os.path.join(save_dir, f"{output_name}_onnx.npy"), output_onnx[0])
            np.save(os.path.join(save_dir, f"{output_name}_torch.npy"), out[0].detach().cpu().numpy())
            print(f"ONNX model input/output saved to {save_dir}")

    print(f"Exported to {os.path.abspath(args.export_llm_path)}")


def _build_minicpm4_7_export_causal_mask(hidden_states, attention_mask):
    """Build the MiniCPM-V 4.7 Qwen3.5-text prefill causal mask.

    Transformers 5.6's ``create_causal_mask`` passes ``inputs_embeds.shape[1]``
    to the selected mask interface.  The legacy ``torch.onnx.export`` tracer
    represents that value as a scalar tensor, while ``sdpa_mask`` interprets
    any tensor there as the old 1-D ``cache_position`` argument and indexes
    element zero.  That combination raises ``IndexError`` during export.

    MiniCPM-V 4.7 multi-segment exports do not carry a past cache: they always
    run the prefill path with positions starting at zero.  The explicit mask
    below is therefore equivalent to the mask returned by the Transformers
    factory for this wrapper, while keeping the sequence dimension dynamic in
    ONNX.
    """
    _, seq_len = hidden_states.shape[:2]
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (seq_len, seq_len), min_dtype, dtype=dtype, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

    # Keep support for callers that already provide a prepared 4-D mask.
    if attention_mask is None or attention_mask.dim() == 4:
        return attention_mask if attention_mask is not None else causal_mask

    if attention_mask.dim() == 2:
        # Use where instead of adding two ``min_dtype`` values.  The latter
        # overflows to ``-inf`` for padded causal positions, while the HF
        # eager mask keeps those entries at the finite ``finfo.min`` value.
        valid_tokens = attention_mask[:, None, None, :].to(torch.bool)
        return torch.where(valid_tokens, causal_mask, torch.full_like(causal_mask, min_dtype))

    return causal_mask

class Qwen3_5SegmentWrapper(torch.nn.Module):
    """Wrap a segment of Qwen3_5DecoderLayer for ONNX export.
    
    Qwen3_5 has mixed layer types (full_attention + linear_attention) and needs
    position_embeddings (cos/sin from mrope) and Tc for linear attention layers.
    
    Inputs match the single-segment export format:
        input_ids, position_ids, attention_mask, Linear_Tc, num_logits_to_keep, conv_state
    """
    def __init__(self, embed_tokens, layers, rotary_emb, norm=None, lm_head=None,
                 is_last_segment=False, layer_indices=None, config=None,
                 use_minicpm4_7_export_mask=False):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        self.norm = norm
        self.lm_head = lm_head
        self.is_last_segment = is_last_segment
        self.layer_indices = layer_indices or list(range(len(layers)))
        self.config = config
        self.use_minicpm4_7_export_mask = use_minicpm4_7_export_mask

    def forward(self, hidden_states, position_ids, attention_mask, Linear_Tc, num_logits_to_keep, conv_state):
        # Embed input_ids to hidden_states (same as single-segment model)
        # hidden_states = self.embed_tokens(input_ids)

        # Compute position embeddings (mrope)
        # position_ids is expected to be (bs, seq_len) for Qwen3_5 mrope
        if position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        cache_position = torch.arange(seq_len, dtype=torch.long, device=device)

        # Only MiniCPM-V 4.7 opts into the explicit export mask. Qwen3.5
        # keeps the original create_causal_mask path, including cache_position.
        if self.use_minicpm4_7_export_mask and (
            torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        ):
            full_attention_mask = _build_minicpm4_7_export_causal_mask(
                hidden_states, attention_mask
            )
        else:
            if self.config is None:
                raise ValueError("Qwen3_5SegmentWrapper requires the Qwen3_5 text config")

            from transformers.masking_utils import create_causal_mask

            full_attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=None,
            )

        # Match Qwen3_5TextModel._update_linear_attn_mask().
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None

        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                layer_mask = linear_attn_mask
            else:
                layer_mask = full_attention_mask

            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                Tc=Linear_Tc,
            )

        if self.is_last_segment:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                # Only keep the last token's logits to match single-segment behavior
                if num_logits_to_keep is not None:
                    hidden_states = self.lm_head(hidden_states.index_select(dim=1, index=num_logits_to_keep))
                else:
                    hidden_states = self.lm_head(hidden_states)
            else:
                print("Warning: lm_head is None in last segment wrapper.")
                exit()
        return hidden_states


def _estimate_module_weight_size(
    module,
    weight_bits,
    group_size=32,
    metadata_bytes_per_group=3,
):
    """Estimate packed weight bytes for a module.

    The default matches RKNN W4A16/W6A16 group32 weights: packed payload plus
    an approximate FP16 scale and 8-bit zero-point for each quantization group.
    Set ``metadata_bytes_per_group`` to zero for non-quantized FP16/FP32 data.
    """
    if module is None:
        return 0

    size = 0
    for parameter in module.parameters():
        numel = parameter.numel()
        size += (numel * weight_bits + 7) // 8
        if group_size and metadata_bytes_per_group:
            size += (
                (numel + group_size - 1) // group_size
            ) * metadata_bytes_per_group
    return size


def _balanced_segment_layer_counts(layer_weight_sizes, num_segments, last_segment_extra_size=0):
    """Split contiguous layers so estimated segment weight sizes are balanced.

    ``last_segment_extra_size`` represents weights that must stay in the final
    segment, such as the final norm and lm_head. Every returned segment contains
    at least one layer, and layer order is preserved.
    """
    num_layers = len(layer_weight_sizes)
    if num_layers == 0:
        return []
    if any(weight_size < 0 for weight_size in layer_weight_sizes):
        raise ValueError("layer weight sizes must be non-negative")
    if last_segment_extra_size < 0:
        raise ValueError("last segment extra size must be non-negative")

    num_segments = min(max(int(num_segments), 1), num_layers)
    if num_segments == 1:
        return [num_layers]

    prefix_weight_sizes = [0]
    for weight_size in layer_weight_sizes:
        prefix_weight_sizes.append(prefix_weight_sizes[-1] + weight_size)

    total_weight_size = prefix_weight_sizes[-1] + last_segment_extra_size

    def scaled_deviation(layer_start, layer_end, segment_index):
        segment_weight_size = (
            prefix_weight_sizes[layer_end] - prefix_weight_sizes[layer_start]
        )
        if segment_index == num_segments:
            segment_weight_size += last_segment_extra_size
        return abs(segment_weight_size * num_segments - total_weight_size)

    # First minimize the largest deviation from the average segment size.
    previous_max_deviations = {0: 0}
    for segment_index in range(1, num_segments + 1):
        current_max_deviations = {}
        min_layer_end = segment_index
        max_layer_end = num_layers - (num_segments - segment_index)

        for layer_end in range(min_layer_end, max_layer_end + 1):
            for layer_start in previous_max_deviations:
                if layer_start >= layer_end:
                    continue
                candidate_max_deviation = max(
                    previous_max_deviations[layer_start],
                    scaled_deviation(layer_start, layer_end, segment_index),
                )
                best_max_deviation = current_max_deviations.get(layer_end)
                if (
                    best_max_deviation is None
                    or candidate_max_deviation < best_max_deviation
                ):
                    current_max_deviations[layer_end] = candidate_max_deviation

        previous_max_deviations = current_max_deviations

    optimal_max_deviation = previous_max_deviations[num_layers]

    # Then, among partitions with that optimal maximum deviation, minimize the
    # total squared deviation. Reverse boundary iteration keeps more blocks in
    # earlier segments when both objectives are exactly tied.
    previous_states = {0: (0, [])}
    for segment_index in range(1, num_segments + 1):
        current_states = {}
        min_layer_end = segment_index
        max_layer_end = num_layers - (num_segments - segment_index)

        for layer_end in range(min_layer_end, max_layer_end + 1):
            for layer_start in sorted(previous_states, reverse=True):
                if layer_start >= layer_end:
                    continue
                deviation = scaled_deviation(layer_start, layer_end, segment_index)
                if deviation > optimal_max_deviation:
                    continue

                previous_squared_deviation, previous_counts = previous_states[layer_start]
                squared_deviation = previous_squared_deviation + deviation * deviation
                best_state = current_states.get(layer_end)
                if best_state is None or squared_deviation < best_state[0]:
                    current_states[layer_end] = (
                        squared_deviation,
                        previous_counts + [layer_end - layer_start],
                    )

        previous_states = current_states

    return previous_states[num_layers][1]


def _resolve_segment_layer_counts(
    layer_weight_sizes,
    num_segments,
    last_segment_extra_size=0,
    segment_layer_counts=None,
):
    """Resolve manual layer counts or fall back to weight-balanced splitting.

    segment_layer_counts is a contiguous partition of the transformer
    layers. When supplied, its length determines the number of segments and
    its values are used verbatim; otherwise the existing weight-balancing
    algorithm is preserved.
    """
    if segment_layer_counts is not None:
        if isinstance(segment_layer_counts, str):
            segment_layer_counts = segment_layer_counts.replace(",", " ").split()
        try:
            counts = [int(count) for count in segment_layer_counts]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--segment_layer_counts must be a list of positive integers"
            ) from exc

        if not counts or any(count <= 0 for count in counts):
            raise ValueError(
                "--segment_layer_counts must contain only positive integers"
            )
        total_layers = len(layer_weight_sizes)
        if sum(counts) != total_layers:
            raise ValueError(
                "--segment_layer_counts must sum to the total number of "
                f"layers ({total_layers}), got {sum(counts)}"
            )
        return counts

    return _balanced_segment_layer_counts(
        layer_weight_sizes,
        num_segments,
        last_segment_extra_size,
    )


def causal_qwen_3_5_llm_to_onnx_multi_segment(
    model, args, use_minicpm4_7_export_mask=False
):
    """Export Qwen3_5 LLM to multiple ONNX files, split by transformer blocks.
    
    Input format matches single-segment causal_qwen_3_5_llm_to_onnx:
        input_ids, position_ids, attention_mask, Linear_Tc, num_logits_to_keep, conv_state
    
    Each segment includes embed_tokens so it can independently embed input_ids.
    The conv_state is split per-segment based on which linear_attention layers
    are in that segment.

    ``use_minicpm4_7_export_mask`` is opt-in so the existing Qwen3.5 export
    path retains its original mask construction.
    """
    import torch
    import os

    # Access text model layers
    text_model = model.model if hasattr(model, 'model') else model
    all_layers = text_model.model.layers
    model_norm = text_model.model.norm if hasattr(text_model.model, 'norm') else None
    embed_tokens = text_model.model.embed_tokens if hasattr(text_model.model, 'embed_tokens') else None
    rotary_emb = text_model.model.rotary_emb if hasattr(text_model.model, 'rotary_emb') else None
    lm_head = text_model.lm_head if hasattr(text_model, 'lm_head') else None

    config = model.config

    num_total_layers = len(all_layers)
    num_segments = getattr(args, 'num_segments', None) or 0

    # Auto-determine segment count if not specified
    if num_segments <= 0:
        layers_per_segment = 8
        num_segments = max(1, (num_total_layers + layers_per_segment - 1) // layers_per_segment)

    # Estimate packed RKNN weights: transformer blocks use W4A16/group32,
    # while the default final lm_head uses W6A16/group32.
    block_weight_bits = 4
    lm_head_weight_bits = 6

    layer_weight_sizes = [
        _estimate_module_weight_size(
            layer,
            block_weight_bits,
        )
        for layer in all_layers
    ]
    last_segment_extra_size = (
        _estimate_module_weight_size(
            model_norm,
            16,
            metadata_bytes_per_group=0,
        )
        + _estimate_module_weight_size(
            lm_head,
            lm_head_weight_bits,
        )
    )
    segment_layer_counts = _resolve_segment_layer_counts(
        layer_weight_sizes,
        num_segments,
        last_segment_extra_size,
        getattr(args, 'segment_layer_counts', None),
    )
    num_segments = len(segment_layer_counts)

    stop_after_segment = getattr(args, 'stop_after_segment', None)
    if stop_after_segment is not None and not 0 <= stop_after_segment < num_segments:
        raise ValueError(
            "--stop_after_segment must be within the exported segment range: "
            f"0 <= {stop_after_segment} < {num_segments}"
        )
    segment_weight_sizes = []
    layer_start = 0
    for segment_index, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        segment_weight_size = sum(layer_weight_sizes[layer_start:layer_end])
        if segment_index == num_segments - 1:
            segment_weight_size += last_segment_extra_size
        segment_weight_sizes.append(segment_weight_size)
        layer_start = layer_end

    print(f"Total layers: {num_total_layers}, Segments: {num_segments}")
    print(f"Layer distribution: {segment_layer_counts}")
    print(
        "Theoretical quantized weight distribution (MiB, INT4 blocks / INT6 lm_head): "
        f"{[round(size / (1024 ** 2), 2) for size in segment_weight_sizes]}"
    )
    print(
        "Last segment fixed norm + INT6 lm_head weight (MiB): "
        f"{last_segment_extra_size / (1024 ** 2):.2f}"
    )
    print(f"Layer types: {config.layer_types}")

    # --- Precompute conv_state shapes for each segment ---
    # Each linear_attention layer has conv_state of shape (d_inner, conv_kernel_size)
    # In single-segment, they are concatenated along dim=0: (num_linear_layers, d_inner, conv_kernel_size)
    # We need to split this per segment
    linear_conv_dim = getattr(config, 'linear_conv_kernel_dim', 4)
    # Calculate d_inner for linear attention (key_dim * 2 + value_dim)
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    k_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    key_dim = num_k_heads * k_head_dim
    value_dim = num_v_heads * v_head_dim
    conv_dim = key_dim * 2 + value_dim  # d_inner for conv1d

    # Count linear_attention layers in each segment
    def count_linear_layers(layer_start, layer_end):
        count = 0
        for i in range(layer_start, layer_end):
            if config.layer_types[i] == "linear_attention":
                count += 1
        return count

    # --- Prepare common args ---
    args.prompt_size = getattr(args, 'prompt_size', 64)
    args.dynamic_shape = getattr(args, 'dynamic_shape', True)
    in_len = args.prompt_size

    model.eval()
    model.float()

    # Base output path
    base_path = args.export_llm_path
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    # --- Build dummy inputs (matching single-segment format) ---
    dummy_input_ids = torch.zeros((1, in_len), dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = embed_tokens(dummy_input_ids)
    attention_mask = torch.ones((1, in_len), dtype=torch.float32)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    Tc = torch.tensor([in_len], dtype=torch.int64)
    num_logits_to_keep = torch.tensor([in_len - 1], dtype=torch.long)

    # Input names must match single-segment format
    input_names = ['input_embeds', 'position_ids', 'attention_mask', 'Linear_Tc', 'num_logits_to_keep', 'conv_state']
    dynamic_axes = {
        'attention_mask': {1: 'seq_len'},
        'position_ids': {1: 'seq_len'},
        'input_embeds': {1: 'seq_len'},
    }

    output_names = ['output']

    # --- Export each segment ---
    layer_start = 0
    for seg_idx, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        is_last = (seg_idx == num_segments - 1)

        segment_layers = list(all_layers[layer_start:layer_end])
        layer_indices = list(range(layer_start, layer_end))

        seg_norm = model_norm if is_last else None
        seg_lm_head = lm_head if is_last else None

        # Build dummy conv_state for this segment
        num_linear = count_linear_layers(layer_start, layer_end)
        dummy_conv_state = torch.zeros((num_linear, conv_dim, linear_conv_dim), dtype=torch.float32)

        segment_model = Qwen3_5SegmentWrapper(
            embed_tokens=embed_tokens,
            layers=segment_layers,
            rotary_emb=rotary_emb,
            norm=seg_norm,
            lm_head=seg_lm_head,
            is_last_segment=is_last,
            layer_indices=layer_indices,
            config=config,
            use_minicpm4_7_export_mask=use_minicpm4_7_export_mask,
        )
        segment_model.eval()
        segment_model.float()

        # Build segment path
        if num_segments == 1:
            seg_path = base_path
        else:
            seg_dir = os.path.join(base_dir, f"seg{seg_idx}")
            os.makedirs(seg_dir, exist_ok=True)
            seg_path = os.path.join(seg_dir, f"{base_name}_seg{seg_idx}.onnx")

        inputs = (inputs_embeds, position_ids, attention_mask, Tc, num_logits_to_keep, dummy_conv_state)

        print(f"Exporting segment {seg_idx}: layers {layer_start}-{layer_end-1} ({layer_count} layers)"
              f", linear_attn={num_linear}"
              f"{' + norm + lm_head' if is_last else ''} -> {seg_path}")

        with torch.no_grad():
            torch.onnx.export(
                segment_model,
                inputs,
                seg_path,
                export_params=True,
                opset_version=19,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                custom_opsets={"rknn": 1},
            )

        print(f"  Segment {seg_idx} exported to {os.path.abspath(seg_path)}")
        layer_start = layer_end
        if stop_after_segment is not None and seg_idx >= stop_after_segment:
            print(
                f"Stopping after segment {seg_idx} as requested by "
                "--stop_after_segment"
            )
            break

    exported_segments = seg_idx + 1
    if stop_after_segment is None:
        print(f"All {num_segments} segments exported successfully.")
    else:
        print(
            f"Exported requested prefix: {exported_segments}/{num_segments} segments."
        )


# disable attribute that may cause error while export onnx
def update_config(_config, _attr_names, _value):
    from transformers import PretrainedConfig

    for _attr in dir(_config):
        if _attr.startswith('__'):
            continue
        try:
            _val = getattr(_config, _attr)
        except Exception:
            continue
        if _attr in _attr_names:
            setattr(_config, _attr, _value)
        elif isinstance(_val, PretrainedConfig):
            update_config(_val, _attr_names, _value)


def export_tokenizer(model_path, tokenizer_path):
    '''Export tokenizer from Hugging Face model to GGUF format.
    Args:
        model_path (str): Path or name of the Hugging Face model.
        tokenizer_path (str): Path to save the exported tokenizer in GGUF format.
    '''
    import subprocess
    import sys

    # remote用于决定是否从远程下载模型文件,如果model_path以'.'、'/'或'~'开头，则remote为0，表示本地文件；否则为1，表示远程文件。
    if model_path.startswith(('.', '/', '~')):
        remote=0
    else:
        remote=1

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    converter = os.path.join(current_dir, '../tokenizer/thirdparty/llama_vocab/convert_hf_to_gguf.py')
    cmd = [sys.executable, converter, '--vocab-only', '--outtype', 'f16', '--outfile', tokenizer_path]
    if remote == 1:
        cmd.append('--remote')
    cmd.append(model_path)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    # 检查命令是否成功执行
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Tokenizer export failed with exit code {result.returncode}")
    else:
        print(f"Tokenizer exported to {tokenizer_path}")


def export_embed_weight(weight, embed_path):
    '''Export embedding weight to float16 .bin.
    Args:
        weight(torch.Tensor): Embedding weight tensor.
        embed_path (str): Path to save the exported embedding weight.
    '''
    import torch

    if not isinstance(weight, torch.Tensor):
        raise TypeError("Weight must be a torch.Tensor")
    
    weight_fp16 = weight.detach().cpu().to(torch.float16).numpy()

    try:
        with open(embed_path, 'wb') as f:
            weight_fp16.tofile(f)
        print(f"Embedding weight exported to {embed_path}")
    except Exception as e:
        print(f"Failed to export embedding weight: {str(e)}")

def split_chat_template_prompt(chat_template, chat_context, prompt="RKLLM"):
    from jinja2 import Template

    tmpl = Template(chat_template)

    chat_template = tmpl.render(**chat_context)

    prompt_prefix, prompt_postfix = chat_template.split(prompt)

    system_prompt = ""
    if 'system' in prompt_prefix:
        sys_idx = prompt_prefix.find("system")
        start_str = prompt_prefix[:sys_idx]
        second_start_idx =  prompt_prefix.find(start_str, sys_idx)
        system_prompt = prompt_prefix[:second_start_idx]

        prompt_prefix = prompt_prefix[second_start_idx:]

    return system_prompt, prompt_prefix, prompt_postfix


def export_internvl_config(model_path, config_path, chat_context=None, prompt="RKLLM"):
    """
      - 读取模型与分词器
      - 用 tokenizer.chat_template 渲染 chat_context（补齐 tools/add_generation_prompt 默认值）
      - 仅拼接文本片段（忽略图片），提取 system/prefix/postfix
      - 汇总并持久化导出配置（含量化可选项）

    参数：
      model_path:  模型路径/名称
      config_path: 导出 pkl 路径
      chat_context: {"messages":[{"role":"user","content": str|list|None}, ...],
                     "tools":[], "add_generation_prompt":bool} 结构
      prompt:      用于切分的最后一次出现的标记（默认 "RKLLM"）
    """
    import pickle
    from copy import deepcopy
    from jinja2 import Template
    from transformers import AutoConfig, AutoTokenizer

    # ---------- 读取配置/分词器 ----------
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---------- 预处理上下文 & 模板渲染（若无模板则直接置空） ----------
    system_prompt = prompt_prefix = prompt_postfix = chat_template = ""
    if getattr(tokenizer, "chat_template", None):
        chat_template = tokenizer.chat_template

        # 规范化 chat_context
        ctx = deepcopy(chat_context) if chat_context is not None else {}
        msgs_in = ctx.get("messages", [])

        def _content_to_str(content):
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(c.get("text", ""))
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            return str(content)

        messages = [{"role": m.get("role", "user"),
                     "content": _content_to_str(m.get("content", ""))}
                    for m in msgs_in]

        rendered = Template(chat_template).render(
            messages=messages,
            tools=ctx.get("tools", []),
            add_generation_prompt=bool(ctx.get("add_generation_prompt", False)),
        )

        # 提取 system（Qwen 风格 <|im_start|>system ... <|im_end|>）
        sys_start, im_end = "<|im_start|>system\n", "<|im_end|>"
        sidx = rendered.find(sys_start)
        if sidx != -1:
            sidx += len(sys_start)
            eidx = rendered.find(im_end, sidx)
            if eidx != -1:
                system_prompt = rendered[sidx:eidx]

        # 按最后一次出现的 prompt 切分
        pidx = rendered.rfind(prompt)
        if pidx == -1:
            prompt_prefix, prompt_postfix = rendered, ""
        else:
            prompt_prefix = rendered[:pidx]
            prompt_postfix = rendered[pidx + len(prompt):]

    # ---------- 汇总导出 ----------
    if not hasattr(config, "llm_config"):
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
    else:
        vocab_size = config.llm_config.vocab_size
        hidden_size = config.llm_config.hidden_size
        
    llm_cfg = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_cfg["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_cfg, f)

    # 可选：打印关键信息（便于调试）
    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    print(f"Model configuration exported to {config_path}")


def export_llm_config(model_path, config_path, chat_context, prompt, user_config=None):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(tokenizer.chat_template )

    if tokenizer.chat_template is not None and chat_context is not None and prompt is not None:
        try:
            system_prompt, prompt_prefix, prompt_postfix = split_chat_template_prompt(tokenizer.chat_template, chat_context, prompt)
        except Exception as e:
            system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
            prompt_prefix  = "<|im_start|>user\n";
            prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
        chat_template = tokenizer.chat_template
    else:
        system_prompt, prompt_prefix, prompt_postfix = "", "", ""
        chat_template = ""

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    vocab_size = text_config.vocab_size
    hidden_size = text_config.hidden_size

    hf_config_json = json.dumps(config.to_dict(), default=str)
    if user_config is not None:
        merged_config = {**config.to_dict(), **user_config}
        hf_config_json = json.dumps(merged_config, default=str)

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "hf_config_json": hf_config_json,
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")



def export_smol_llm_config(model_path, config_path):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("chat_template:\n", repr(tokenizer.chat_template)[1:-1])

    system_prompt = "<|im_start|>System: You are a useful assistant for concise replies.<end_of_utterance>\n"
    prompt_prefix = "User: "
    prompt_postfix = "<end_of_utterance>\nAssistant:"
    chat_template = ""
    
    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    vocab_size = text_config.vocab_size
    hidden_size = text_config.hidden_size

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")


def export_janus_pro_llm_config(model_path, grq_model_path, config_path, conversation, prompt="RKLLM"):
    from janus.models import VLChatProcessor
    from transformers import AutoConfig
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    text_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, system_prompt=VLChatProcessor.system_prompt,
    )

    assert "<|User|>" in text_prompt, "Janus_Pro conversation not include <|User|> "

    system_prompt, _ = text_prompt.split("<|User|>")
    prompt_idx = text_prompt.find("<|User|>")
    prompt_prefix, prompt_postfix = text_prompt[prompt_idx:].split(prompt)

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": "" if tokenizer.chat_template is None else tokenizer.chat_template,
        "vocab_size": config.language_config.vocab_size,
        "hidden_size": config.language_config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    grq_config = AutoConfig.from_pretrained(grq_model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")

def export_minicpm_3o_llm_config(model_path, grq_model_path, config_path, message, prompt="RKLLM"):
    from PIL import Image
    from transformers import AutoProcessor, AutoConfig
    import json
    from copy import deepcopy
    import pickle
    import re

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    msgs_list = [message]
    images_list = [None]
    prompts_lists = []
    input_images_lists = []
    for image, msgs in zip(images_list, msgs_list):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)
        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)
        prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
        input_images_lists.append(images)

    pattern = "(<image>./</image>)"
    
    final_texts = []
    for index, text in enumerate(prompts_lists):
        image_tags = re.findall(pattern, text)
        text_chunks = text.split(pattern)
        final_text = ""
        for i in range(len(image_tags)):
            final_text = final_text + text_chunks[i] + processor.image_processor.get_slice_image_placeholder(input_images_lists[i][0].size, i,)
        final_text += text_chunks[-1]
        final_texts.append(final_text)

    system_prompt = ""
    prompt_prefix, prompt_postfix = final_texts[0].split(prompt)
    if 'system' in prompt_prefix:
        sys_idx = prompt_prefix.find("system")
        start_str = prompt_prefix[:sys_idx]
        second_start_idx =  prompt_prefix.find(start_str, sys_idx)
        system_prompt = prompt_prefix[:second_start_idx]

        prompt_prefix = prompt_prefix[second_start_idx:]

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": processor.tokenizer.chat_template,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    grq_config = AutoConfig.from_pretrained(grq_model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")

def export_minicpm_v_llm_config(model_path, config_path, conversation, prompt="RKLLM"):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # pip install peft

    # 来自modeling_minicpmv中MiniCPMV类的chat方法
    text_prompt = ''
    for i, msg in enumerate(conversation):
        role = msg['role']
        content = msg['content']
        assert role in ['user', 'assistant']
        if i == 0:
            assert role == 'user', 'The role of first msg should be user'
            content = tokenizer.im_start + tokenizer.unk_token * config.query_num + tokenizer.im_end + '\n' + content
        text_prompt += '<用户>' if role=='user' else '<AI>'
        text_prompt += content
    text_prompt += '<AI>'

    system_prompt = ""
    prompt_prefix, prompt_postfix = text_prompt.split(prompt)

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": tokenizer.chat_template,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    if hasattr(config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': config.quantization_config['bits'],
            'sym': config.quantization_config['sym'],
            'group_size': config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")
