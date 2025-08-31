import warnings

warnings.filterwarnings("ignore")
import os
import pickle
import argparse
import time
from typing import Optional

import cProfile
import pstats
import io

import torch
import torch._inductor.config
import torch._functorch.config
import torch.nn.functional as F
import tiktoken
import numpy as np
import onnxruntime
from html2term import printc

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

from model import GPTConfig

from export_utils import create_onnx_model_for_inference

def _apply_sampling(
    logits: torch.Tensor, temp: float, top_p: Optional[float], top_k: Optional[int]
) -> int:
    """Apply temperature, top-p, and top-k sampling to logits, expecting a GPU tensor."""
    if temp == 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits.div_(temp)

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")

    elif top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1, None]] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def run_chat_loop_io_binding(onnx_model_path, config, tokenizer, device, args):
    """Runs a highly optimized interactive chat loop using I/O Binding to keep data on the GPU."""
    if not args.profile:
        printc("<br/><blue>--- Starting Chat with Fused FP16 ONNX Model (I/O Binding Enabled) ---</blue>")
        sampling_params = f"temp={args.temperature}"
        if args.temperature == 0: sampling_params = "greedy"
        elif args.top_p is not None: sampling_params += f", top_p={args.top_p}"
        elif args.top_k is not None: sampling_params += f", top_k={args.top_k}"
        printc(f"<#cccccc>Params: {sampling_params}, max_new_tokens={args.max_new_tokens}. Type 'exit' or 'quit' to end.</#cccccc>")

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    
    base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    optimized_model_path = os.path.join(os.path.dirname(onnx_model_path), f"{base_name}.ort")
    options.optimized_model_filepath = optimized_model_path
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    if device == "cuda":
        provider = "CUDAExecutionProvider"
        if provider not in onnxruntime.get_available_providers():
            raise RuntimeError(f"{provider} not available, please check your ONNX Runtime and CUDA setup.")
        device_id = torch.cuda.current_device()
        provider_options = {'device_id': device_id, 'arena_extend_strategy': 'kSameAsRequested'}
        providers = [
            ('CUDAExecutionProvider', provider_options),
            'CPUExecutionProvider',
        ]
        if not args.profile: printc(f"Using ONNX Runtime providers: <b>{providers[0][0]} (with CUDA Graph), {providers[1]}</b>")
        device_name = 'cuda'
        torch_device = torch.device(f"cuda:{device_id}")
    else:
        provider = "CPUExecutionProvider"
        if not args.profile: printc(f"Using ONNX Runtime provider: <b>{provider}</b>")
        providers.append(provider)
        device_name = 'cpu'
        torch_device = torch.device("cpu")

    session = onnxruntime.InferenceSession(onnx_model_path, sess_options=options, providers=providers)
    stop_ids = [tokenizer.encode_single_token(t) for t in ["<|endoftext|>", "<|user|>"]]
    
    conversation_history_ids = []

    while True:
        try:
            if args.profile:
                prompt = "What is the capital of France and what is its history?"
                printc(f"<br/><b>You: </b>{prompt}")
            else:
                printc("<br/><b>You: </b>", end="")
                prompt = input()
            
            if prompt.lower() in ["exit", "quit"]: break

            if not conversation_history_ids:
                tokens_to_process = tokenizer.encode(f"<|startoftext|><|user|>{prompt}<|assistant|>", allowed_special="all")
            else:
                tokens_to_process = tokenizer.encode(f"<|user|>{prompt}<|assistant|>", allowed_special="all")

            conversation_history_ids.extend(tokens_to_process)

            if len(conversation_history_ids) > config.block_size:
                printc("<br/><red>[CONTEXT RESET - Model has forgotten the conversation]</red>")
                conversation_history_ids = tokenizer.encode(f"<|startoftext|><|user|>{prompt}<|assistant|>", allowed_special="all")
                tokens_to_process = conversation_history_ids

            binding = session.io_binding()
            
            input_ids_np = np.array([tokens_to_process], dtype=np.int64)
            input_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids_np, device_name, 0)
            binding.bind_ortvalue_input('input_ids', input_ids_ort)

            dtype = np.float16
            empty_past = np.zeros((1, config.n_kv_heads, 0, config.n_embd // config.n_heads), dtype=dtype)
            empty_past_ort = onnxruntime.OrtValue.ortvalue_from_numpy(empty_past, device_name, 0)
            for i in range(config.n_layers):
                binding.bind_ortvalue_input(f'past_key_{i}', empty_past_ort)
                binding.bind_ortvalue_input(f'past_value_{i}', empty_past_ort)

            binding.bind_output('logits', device_name)
            for i in range(config.n_layers):
                binding.bind_output(f'present_key_{i}', device_name)
                binding.bind_output(f'present_value_{i}', device_name)

            session.run_with_iobinding(binding)
            ort_outs = binding.get_outputs()
            logits_ort, past_key_values = ort_outs[0], ort_outs[1:]

            logits_torch = torch.tensor(logits_ort.numpy(), device="cuda")
            next_token_id = _apply_sampling(logits_torch[0, -1, :], args.temperature, args.top_p, args.top_k)

            printc("<b>Bot: </b>", end="", flush=True)
            generated_response_ids = []
            start_time = time.perf_counter()

            max_tokens = min(args.max_new_tokens, config.block_size - len(conversation_history_ids))
            if max_tokens <= 0:
                printc("<br/><yellow>[CONTEXT FULL - Cannot generate more tokens]</yellow>")
                if args.profile: break
                continue
            
            single_token_input_ort = onnxruntime.OrtValue.ortvalue_from_numpy(
                np.array([[next_token_id]], dtype=np.int64), device_name, 0
            )

            for _ in range(max_tokens):
                if next_token_id in stop_ids: break

                generated_response_ids.append(next_token_id)
                print(tokenizer.decode([next_token_id]), end="", flush=True)
                
                binding.bind_ortvalue_input('input_ids', single_token_input_ort)
                for j in range(config.n_layers):
                    binding.bind_ortvalue_input(f'past_key_{j}', past_key_values[j*2])
                    binding.bind_ortvalue_input(f'past_value_{j}', past_key_values[j*2+1])
                
                binding.bind_output('logits', device_name)
                for j in range(config.n_layers):
                    binding.bind_output(f'present_key_{j}', device_name)
                    binding.bind_output(f'present_value_{j}', device_name)

                session.run_with_iobinding(binding)
                ort_outs = binding.get_outputs()
                
                logits_ort, past_key_values = ort_outs[0], ort_outs[1:]

                logits_torch = torch.tensor(logits_ort.numpy(), device="cuda")
                next_token_id = _apply_sampling(logits_torch[0, 0, :], args.temperature, args.top_p, args.top_k)
                
                single_token_input_ort.update_inplace(np.array([[next_token_id]], dtype=np.int64))


            end_time = time.perf_counter()
            conversation_history_ids.extend(generated_response_ids)
            printc("<br/>")

            num_generated = len(generated_response_ids)
            time_taken = end_time - start_time
            tokens_per_sec = num_generated / time_taken if time_taken > 0 else 0
            printc(f"<#cccccc><i>Generated {num_generated} tokens in {time_taken:.2f}s ({tokens_per_sec:.2f} tokens/s)</i></#cccccc>")
            
            if args.profile:
                break

        except KeyboardInterrupt:
            printc("<br/><yellow>Exiting chat mode.</yellow>")
            break
        except Exception as e:
            printc(f"<br/><red>An error occurred: {e}</red>")
            import traceback
            traceback.print_exc()
            break

def main():
    parser = argparse.ArgumentParser(
        description="Optimize a GPT model to Fused FP16 ONNX and run a fast chat session."
    )
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_ft/best_model.pt", help="Path to the PyTorch model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for model loading and ONNX export.")
    parser.add_argument("--max_new_tokens", type=int, default=480, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature. 0 for greedy.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling.")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile to analyze performance of one generation cycle.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        printc("<red>CUDA is selected but not available. Please check your environment.</red>")
        exit(1)

    if args.temperature == 0:
        printc("<yellow>Temperature is 0, using greedy decoding.</yellow>")

    ONNX_MODEL_DIR = "onnx_models"
    os.makedirs(ONNX_MODEL_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    onnx_fp16_fused_path = os.path.join(ONNX_MODEL_DIR, f"{base_name}_fp16_kv_fused.onnx")

    with open("tokenizer/Hastings.pkl", "rb") as f:
        hastings = pickle.load(f)
    enc = tiktoken.core.Encoding(hastings.pop("name"), **hastings)
    
    create_onnx_model_for_inference(args.checkpoint_path, onnx_fp16_fused_path, args.device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    del checkpoint

    if args.profile:
        printc("<bg-magenta><black> --- PROFILING MODE ENABLED --- </black></bg-magenta>")
        pr = cProfile.Profile()
        pr.enable()
        run_chat_loop_io_binding(onnx_fp16_fused_path, config, enc, args.device, args)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats(30)
        printc("<br/><magenta>--- Profiler Results (Top 30 by Cumulative Time) ---</magenta>")
        print(s.getvalue())
    else:
        run_chat_loop_io_binding(onnx_fp16_fused_path, config, enc, args.device, args)


if __name__ == "__main__":
    main()
