import os
import argparse
import torch
from html2term import printc

from tokenizer_hf import create_tokenizer_from_custom_pickle
from model_hf import LilleConfig, LilleForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained GPT model to a local directory in Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../checkpoints_ft/best_model.pt",
        help="Path to the PyTorch model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the Hugging Face compatible model files will be saved.",
    )
    parser.add_argument(
        "--tokenizer_pickle_path",
        type=str,
        default="../tokenizer/Hastings.pkl",
        help="Path to the custom tokenizer pickle file.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = create_tokenizer_from_custom_pickle(args.tokenizer_pickle_path)

    printc(f"<b>Loading checkpoint from:</b> <i>{args.checkpoint_path}</i>")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if not model_args:
        printc("<red>Error: 'model_args' not found in the checkpoint.</red>")
        return

    printc(f"<b>Using vocab size from checkpoint:</b> {model_args['vocab_size']}")
    printc(
        f"<b>Custom tokenizer length is:</b> {len(tokenizer)} (This should be consistent with the checkpoint)"
    )

    if "n_layers" in model_args:
        model_args["n_layer"] = model_args.pop("n_layers")
    if "n_heads" in model_args:
        model_args["n_head"] = model_args.pop("n_heads")

    config = LilleConfig(**model_args)
    model = LilleForCausalLM(config)

    printc("<b>Loading model weights...</b>")
    state_dict = checkpoint["model_state_dict"]

    if config.tie_word_embeddings:
        if "lm_head.weight" in state_dict:
            printc(
                "<b>NOTE:</b> <yellow>Removing 'lm_head.weight' from state_dict due to tied weights.</yellow>"
            )
            state_dict.pop("lm_head.weight")

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.transformer.load_state_dict(state_dict, strict=True)
    model.eval()
    printc(
        f"<green>Model loaded successfully with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.</green>"
    )

    chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>' + message['content'] + '<|assistant|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    tokenizer.chat_template = chat_template

    printc(f"<br/><b>Saving model and tokenizer to:</b> <cyan>{args.output_dir}</cyan>")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    printc(f"\n<b><green>✅ Export complete!</green></b>")


if __name__ == "__main__":
    main()
