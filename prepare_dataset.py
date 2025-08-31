import numpy as np
import os
from tqdm import tqdm
import tiktoken
import pickle
import gc
import random
from html2term import printc

dataset = "smol-sft"
input_file_path = f"data/{dataset}/train.txt"
output_dir = f"data/{dataset}"
val_split = 0.1
EOT_TOKEN = "<|endoftext|>"
TOKENIZER_PATH = 'tokenizer/Hastings.pkl'

def load_tokenizer(tokenizer_path=TOKENIZER_PATH):
    """Loads the tokenizer and returns the encoding instance."""
    with open(tokenizer_path, 'rb') as f:
        hastings = pickle.load(f)
    enc = tiktoken.core.Encoding(hastings.pop('name'), **hastings)
    printc(f"Tokenizer loaded. Vocab size: <b>{enc.n_vocab}</b>")
    try:
        enc.encode_single_token(EOT_TOKEN)
    except KeyError:
        raise ValueError(f"The EOT token '{EOT_TOKEN}' is not in the tokenizer vocabulary!")
    return enc

def process_and_save(lines, enc, output_path, split_name):
    """Tokenizes lines and saves them in the efficient tokens/offsets format."""
    printc(f"<br/><b>Processing and tokenizing {len(lines):,} documents for the <i>{split_name}</i> split...</b>")

    all_tokenized = []
    for doc in tqdm(lines, desc=f"Tokenizing {split_name}"):
        if doc:
            tokens = enc.encode(doc, allowed_special='all')
            all_tokenized.append(tokens)

    total_tokens = sum(len(doc) for doc in all_tokenized)
    num_docs = len(all_tokenized)

    tokens_arr = np.empty(total_tokens, dtype=np.uint16)
    offsets_arr = np.empty(num_docs + 1, dtype=np.uint64)
    offsets_arr[0] = 0

    token_pos = 0
    for i, doc in enumerate(all_tokenized):
        doc_len = len(doc)
        tokens_arr[token_pos : token_pos + doc_len] = doc
        token_pos += doc_len
        offsets_arr[i + 1] = token_pos

    del all_tokenized
    gc.collect()

    printc(f"  Saving <i>{split_name}</i> data to <i>{output_path}</i>...")
    np.savez_compressed(output_path, tokens=tokens_arr, offsets=offsets_arr)
    printc(f"  <green>Saved {num_docs:,} documents ({total_tokens:,} tokens).</green>")

def main():
    os.makedirs(output_dir, exist_ok=True)
    enc = load_tokenizer()

    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    docs = content.split(EOT_TOKEN)
    documents = [doc.strip() + EOT_TOKEN for doc in docs if doc.strip()]
    printc(f"Found and processed <b>{len(documents):,}</b> documents.")
    
    random.seed(42)
    random.shuffle(documents)
    printc("<i>Shuffled documents randomly to ensure a representative validation split.</i>")
    
    split_idx = int(len(documents) * (1 - val_split))
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]

    train_output_path = os.path.join(output_dir, "train.npz")
    val_output_path = os.path.join(output_dir, "val.npz")

    process_and_save(train_docs, enc, train_output_path, "train")
    process_and_save(val_docs, enc, val_output_path, "validation")

    printc("<br/><b><green>✅ Processing complete.</green></b>")

if __name__ == "__main__":
    main()
