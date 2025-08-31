import os
import pickle
import json
import tempfile

from html2term import printc
from transformers import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

def create_tokenizer_from_custom_pickle(pickle_path: str) -> GPT2Tokenizer:
    printc(f"<b>Building tokenizer from custom file:</b> <i>{pickle_path}</i>")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Custom tokenizer pickle file not found at {pickle_path}")

    with open(pickle_path, 'rb') as f:
        hastings_data = pickle.load(f)

    byte_decoder = bytes_to_unicode()
    final_token_to_rank = {}
    
    for token_bytes, rank in hastings_data['mergeable_ranks'].items():
        token_str = "".join([byte_decoder[b] for b in token_bytes])
        final_token_to_rank[token_str] = rank
        
    for token_str, rank in hastings_data['special_tokens'].items():
        final_token_to_rank[token_str] = rank

    valid_token_strings = set(final_token_to_rank.keys())

    sorted_vocab = sorted(final_token_to_rank.items(), key=lambda item: item[1])
    final_vocab_dict = {token: i for i, (token, rank) in enumerate(sorted_vocab)}

    printc("<b>Reconstructing BPE merges based on the new vocabulary...</b>")
    base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
    
    re_ranked_merges = []
    for pair, original_rank in base_tokenizer.bpe_ranks.items():
        p1, p2 = pair
        merged_token_str = p1 + p2

        if p1 in valid_token_strings and p2 in valid_token_strings and merged_token_str in valid_token_strings:
            new_rank = final_token_to_rank[merged_token_str]
            re_ranked_merges.append((new_rank, pair))

    re_ranked_merges.sort(key=lambda x: x[0])
    
    final_merges_formatted = [f"{p1} {p2}" for rank, (p1, p2) in re_ranked_merges]
    
    printc(f"<b>Target vocab size:</b> {hastings_data['explicit_n_vocab']}. <b>Reconstructed valid merges:</b> {len(final_merges_formatted)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        vocab_path = os.path.join(temp_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(final_vocab_dict, f, ensure_ascii=False)
            
        merges_path = os.path.join(temp_dir, 'merges.txt')
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            f.write("\n".join(final_merges_formatted))

        config_path = os.path.join(temp_dir, 'tokenizer_config.json')
        tokenizer_config = {
            "model_max_length": 512,
            "add_prefix_space": True,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False)

        tokenizer = GPT2Tokenizer.from_pretrained(temp_dir)

    special_tokens_map = {
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": ["<|assistant|>", "<|user|>"]
    }
    tokenizer.add_special_tokens(special_tokens_map)
    printc("<green>Custom tokenizer built successfully!</green>")
    return tokenizer
