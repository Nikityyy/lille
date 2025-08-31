import os
import sys
import torch
import onnx
import onnxsim
from onnxruntime.transformers import optimizer
from html2term import printc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig

def load_pytorch_model(checkpoint_path, device="cpu", use_fp16=False):
    """
    Loads the PyTorch model from a checkpoint and optionally converts to FP16.
    """
    printc(f"<b>Loading PyTorch checkpoint from:</b> <i>{checkpoint_path}</i>")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint["model_state_dict"]

    unwanted_prefixes = ["_orig_mod.", "module."]
    for k, v in list(state_dict.items()):
        for prefix in unwanted_prefixes:
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = state_dict.pop(k)
                break
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if use_fp16:
        printc("<b>Converting PyTorch model to FP16...</b>")
        model.half()

    model.to(device)
    printc("<green>PyTorch model loaded successfully.</green>")
    return model, config


def export_unified_onnx_model(model, onnx_path, device):
    """Exports a single, unified ONNX model for both prefill and decode."""
    printc(f"<b>Exporting UNIFIED model to ONNX (FP16) at:</b> <i>{onnx_path}</i>")
    config = model.config
    
    input_names = ["input_ids"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    model_dtype = torch.float16
    dummy_past_kv = []
    for i in range(config.n_layers):
        past_key, past_val = f"past_key_{i}", f"past_value_{i}"
        present_key, present_val = f"present_key_{i}", f"present_value_{i}"
        
        input_names.extend([past_key, past_val])
        output_names.extend([present_key, present_val])
        
        dynamic_axes.update({
            past_key: {0: "batch_size", 2: "past_sequence_len"},
            past_val: {0: "batch_size", 2: "past_sequence_len"},
            present_key: {0: "batch_size", 2: "total_sequence_len"},
            present_val: {0: "batch_size", 2: "total_sequence_len"},
        })
        
        dummy_past_kv.append((
            torch.randn(1, config.n_kv_heads, 12, config.n_embd // config.n_heads, device=device, dtype=model_dtype),
            torch.randn(1, config.n_kv_heads, 12, config.n_embd // config.n_heads, device=device, dtype=model_dtype),
        ))

    dummy_input_ids = torch.ones(1, 1, dtype=torch.long, device=device)
    model_args = (dummy_input_ids, dummy_past_kv, True)

    torch.onnx.export(
        model,
        model_args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )
    printc("<green>Unified ONNX export complete.</green>")


def simplify_and_optimize_onnx(unsimplified_path, final_path, config):
    """Simplifies and optimizes a single ONNX model."""
    printc(f"<b>Simplifying and optimizing:</b> <i>{unsimplified_path}</i>")
    temp_simplified_path = unsimplified_path.replace(".onnx", "_simplified.onnx")
    
    onnx_model = onnx.load(unsimplified_path)
    model_simplified, check = onnxsim.simplify(onnx_model)
    if not check:
        printc("<bg-red><white>ONNX simplification failed. Using unsimplified model.</white></bg-red>")
        onnx.save(onnx_model, temp_simplified_path)
    else:
        onnx.save(model_simplified, temp_simplified_path)

    opt_model = optimizer.optimize_model(
        input=temp_simplified_path,
        model_type="gpt2",
        num_heads=config.n_heads,
        hidden_size=config.n_embd,
        opt_level=2,
        use_gpu=True,
        only_onnxruntime=False,
    )
    
    printc("<b>Converting optimized model to FP16...</b>")
    opt_model.convert_model_float32_to_float16()

    opt_model.save_model_to_file(final_path)
    printc(f"<green>Optimization complete. Final model at: <i>{final_path}</i></green>")
    
    if os.path.exists(temp_simplified_path):
        os.remove(temp_simplified_path)


def create_onnx_model_for_inference(
    checkpoint_path, onnx_model_path, device="cuda"
):
    """Full pipeline to create a final, optimized ONNX model."""
    if os.path.exists(onnx_model_path):
        printc("<bg-yellow><black>Final ONNX model already exists, skipping generation.</black></bg-yellow>")
        _, config = load_pytorch_model(checkpoint_path, device, use_fp16=False)
        return config

    temp_unsimplified_path = onnx_model_path.replace(".onnx", "_temp_unsimplified.onnx")

    try:
        model, config = load_pytorch_model(checkpoint_path, device, use_fp16=True)
        
        export_unified_onnx_model(model, temp_unsimplified_path, device)
        
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

        simplify_and_optimize_onnx(temp_unsimplified_path, onnx_model_path, config)

        return config

    finally:
        if os.path.exists(temp_unsimplified_path):
            os.remove(temp_unsimplified_path)
            printc(f"<i>Cleaned up temporary file: {temp_unsimplified_path}</i>")
