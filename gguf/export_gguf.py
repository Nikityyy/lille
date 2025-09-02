import os
import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from simple_ai.model_hf import LilleConfig, LilleForCausalLM
from typing import Iterable
from torch import Tensor

LLAMA_CPP_DIR = "llama.cpp"
if not os.path.isdir(LLAMA_CPP_DIR):
    print(f"'{LLAMA_CPP_DIR}' directory not found.")
    print("Attempting to clone the repository...")
    LLAMA_CPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", LLAMA_CPP_REPO_URL, LLAMA_CPP_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Successfully cloned '{LLAMA_CPP_REPO_URL}' into '{LLAMA_CPP_DIR}'.")
    except FileNotFoundError:
        print("Error: 'git' command not found.")
        print(
            "Please install Git and ensure it is in your system's PATH to automatically clone llama.cpp."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to clone the llama.cpp repository.")
        print(f"Git command failed with exit code {e.returncode}.")
        print(f"Stderr: {e.stderr.strip()}")
        sys.exit(1)

sys.path.insert(0, LLAMA_CPP_DIR)

try:
    from convert_hf_to_gguf import main as convert_main, ModelBase, gguf, LlamaModel  # type: ignore
except ImportError as e:
    print(
        f"Error: Failed to import from {os.path.join(LLAMA_CPP_DIR, 'convert_hf_to_gguf.py')}: {e}"
    )
    print("Please ensure your llama.cpp clone is up-to-date and the file exists.")
    sys.exit(1)


# 1. Define and Register the GGUF Conversion Class ---
@ModelBase.register("LilleForCausalLM")
class LilleModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_context_length(self.hparams["block_size"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["n_kv_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_eps"])
        self.gguf_writer.add_rope_freq_base(self.hparams["rope_theta"])

        multiple_of = 256
        n_embd = self.hparams["n_embd"]
        ff_dim = int(2 * (4 * n_embd) / 3)
        ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.gguf_writer.add_feed_forward_length(ff_dim)

    def set_vocab(self):
        tokens: list[str] = []
        toktypes: list[int] = []

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)

        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = "gpt-2"

        reverse_vocab = {
            id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()
        }
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if added_tokens_decoder[i].special or self.does_token_look_special(
                        token
                    ):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.endswith((".cos_cached", ".sin_cached")):
            return

        n_head = self.hparams["n_head"]
        n_kv_head = self.hparams["n_kv_heads"]

        if name.endswith("attention.qkv_proj.weight"):
            n_embd = self.hparams["n_embd"]
            head_dim = n_embd // n_head

            q_size = n_head * head_dim
            k_size = n_kv_head * head_dim
            v_size = n_kv_head * head_dim

            q_proj, k_proj, v_proj = torch.split(
                data_torch, [q_size, k_size, v_size], dim=0
            )

            q_proj = LlamaModel.permute(q_proj, n_head, n_head)
            k_proj = LlamaModel.permute(k_proj, n_head, n_kv_head)

            yield self.map_tensor_name(
                f"model.layers.{bid}.self_attn.q_proj.weight"
            ), q_proj
            yield self.map_tensor_name(
                f"model.layers.{bid}.self_attn.k_proj.weight"
            ), k_proj
            yield self.map_tensor_name(
                f"model.layers.{bid}.self_attn.v_proj.weight"
            ), v_proj
            return

        rename_map = {
            "transformer.tok_embeddings.weight": "model.embed_tokens.weight",
            "transformer.norm.weight": "model.norm.weight",
            "transformer.output.weight": "lm_head.weight",
            f"transformer.layers.{bid}.attention.out_proj.weight": f"model.layers.{bid}.self_attn.o_proj.weight",
            f"transformer.layers.{bid}.attention.norm.weight": f"model.layers.{bid}.input_layernorm.weight",
            f"transformer.layers.{bid}.feed_forward.gate_proj.weight": f"model.layers.{bid}.mlp.gate_proj.weight",
            f"transformer.layers.{bid}.feed_forward.up_proj.weight": f"model.layers.{bid}.mlp.up_proj.weight",
            f"transformer.layers.{bid}.feed_forward.down_proj.weight": f"model.layers.{bid}.mlp.down_proj.weight",
            f"transformer.layers.{bid}.feed_forward.norm.weight": f"model.layers.{bid}.post_attention_layernorm.weight",
        }

        if name in rename_map:
            translated_name = rename_map[name]
            yield self.map_tensor_name(translated_name), data_torch
            return

        raise ValueError(f"Can not map tensor {name!r}")


# 2. Register the custom HF model architecture
AutoConfig.register("lille-130m", LilleConfig)
AutoModelForCausalLM.register(LilleConfig, LilleForCausalLM)

# 3. Define constants
MODEL_NAME = "Nikity/lille-130m-instruct"
LOCAL_MODEL_DIR = "./lille-130m-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Define quantization types
QUANTIZATION_TYPES = ["q8_0", "f16", "f32"]

# 5. Download and save the model and tokenizer
print("Downloading and saving model...")
try:
    if not os.path.exists(LOCAL_MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map=DEVICE
        )
        model.save_pretrained(LOCAL_MODEL_DIR)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        print(f"Model and tokenizer saved to {LOCAL_MODEL_DIR}")
    else:
        print(f"Model already exists at {LOCAL_MODEL_DIR}. Skipping download.")
except Exception as e:
    print(f"Failed to download/save model: {e}")
    raise

# 6. Inspect model configuration (for debugging)
print("Model configuration:")
config = AutoConfig.from_pretrained(LOCAL_MODEL_DIR)
print(config)

# 7. Convert to GGUF for each quantization type
for quant_type in QUANTIZATION_TYPES:
    out_dir = "gguf_models"
    os.makedirs(out_dir, exist_ok=True)
    output_gguf = os.path.join(out_dir, f"lille-130m-instruct-{quant_type}.gguf")
    print(f"Attempting to convert to GGUF with quantization {quant_type}...")

    # Build argument list
    command_args = [LOCAL_MODEL_DIR, "--outfile", output_gguf, "--outtype", quant_type]

    original_argv = sys.argv
    try:
        sys.argv = ["convert_hf_to_gguf.py"] + command_args

        convert_main()

        print(f"Conversion successful! GGUF file created at {output_gguf}")
    except Exception as e:
        print(f"GGUF conversion failed for {quant_type}: {e}")
        raise
    finally:
        sys.argv = original_argv
