import os
import re
import json
import gc
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
from html2term import printc

dataset_name = "fineweb_edu_sample_10BT"
output_dir = f"data/{dataset_name}"
temp_dir = os.path.join(output_dir, "tmp")
state_file = os.path.join(temp_dir, "progress.json")
TOKENIZER_PATH = "tokenizer/Hastings.pkl"

hf_dataset_name = "HuggingFaceFW/fineweb-edu"
hf_dataset_config = "sample-10BT"
hf_dataset_split = "train"
num_documents = 9_672_101

CHUNK_SIZE = 200_000
language_filter = "en"
language_score_threshold = 0.95
val_split = 0.0005
EOT_TOKEN = "<|endoftext|>"
REMOVE_CHUNKS_AFTER_CONSOLIDATION = True

def load_tokenizer(tokenizer_path=TOKENIZER_PATH):
    """Loads the tokenizer from a pickle file."""
    with open(tokenizer_path, "rb") as f:
        hastings = pickle.load(f)
    enc = tiktoken.core.Encoding(hastings.pop("name"), **hastings)
    printc(f"Tokenizer loaded. Vocab size: <b>{enc.n_vocab}</b>")
    try:
        enc.encode_single_token(EOT_TOKEN)
    except KeyError:
        raise ValueError(f"EOT token '{EOT_TOKEN}' is not in the tokenizer vocabulary.")
    return enc

def load_progress():
    """Loads the processing progress from the state file."""
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            p = json.load(f)
            return p.get("processed_docs", 0), p.get("chunk_index", 0)
    return 0, 0

def save_progress(processed_docs, chunk_index):
    """Saves the current processing progress."""
    os.makedirs(temp_dir, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump({"processed_docs": processed_docs, "chunk_index": chunk_index}, f)

def save_chunk(docs, chunk_index, split_name):
    """Saves a chunk of tokenized documents to a temporary file."""
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{split_name}_chunk_{chunk_index}.npy")
    np.save(file_path, np.array(docs, dtype=object))

def _get_chunk_files(temp_dir, split_name):
    """Helper to list and sort temporary chunk files."""
    pattern = re.compile(rf"^{re.escape(split_name)}_chunk_(\d+)\.npy$")
    files = []
    if not os.path.isdir(temp_dir):
        return []
    for fn in os.listdir(temp_dir):
        m = pattern.match(fn)
        if m:
            idx = int(m.group(1))
            files.append((idx, os.path.join(temp_dir, fn)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]

def consolidate_chunks_to_npz(temp_dir, output_dir, split_name, remove_chunks=REMOVE_CHUNKS_AFTER_CONSOLIDATION):
    """Consolidates temporary chunks into a final compressed .npz file."""
    chunk_files = _get_chunk_files(temp_dir, split_name)
    if not chunk_files:
        printc(f"<yellow>No chunk files found for split '<i>{split_name}</i>'. Skipping.</yellow>")
        return

    printc(f"<b>Pass 1/2:</b> Counting tokens in <b>{len(chunk_files)}</b> chunks for '<i>{split_name}</i>'...")
    total_docs = 0
    total_tokens = 0
    for p in tqdm(chunk_files, desc=f"Counting {split_name}"):
        arr = np.load(p, allow_pickle=True)
        total_docs += arr.shape[0]
        total_tokens += sum(len(x) for x in arr)
        del arr
    gc.collect()

    if total_docs == 0:
        printc(f"<yellow>No documents found in chunks for '<i>{split_name}</i>'. Skipping.</yellow>")
        return

    printc(f"  Found <b>{total_docs:,}</b> documents and <b>{total_tokens:,}</b> tokens for '<i>{split_name}</i>'.")

    tokens_arr = np.empty(total_tokens, dtype=np.uint16)
    offsets_arr = np.empty(total_docs + 1, dtype=np.uint64)
    offsets_arr[0] = 0

    printc(f"<b>Pass 2/2:</b> Consolidating chunks into final arrays...")
    token_pos = 0
    doc_idx = 0
    for p in tqdm(chunk_files, desc=f"Consolidating {split_name}"):
        chunk = np.load(p, allow_pickle=True)
        for doc in chunk:
            doc_len = len(doc)
            tokens_arr[token_pos : token_pos + doc_len] = doc
            token_pos += doc_len
            doc_idx += 1
            offsets_arr[doc_idx] = token_pos
        del chunk
    gc.collect()

    output_path = os.path.join(output_dir, f"{split_name}.npz")
    printc(f"  Saving consolidated data to <i>{output_path}</i>...")
    np.savez_compressed(output_path, tokens=tokens_arr, offsets=offsets_arr)
    printc(f"  <green>Successfully saved <i>{output_path}</i></green>")

    if remove_chunks:
        printc("  Cleaning up temporary chunk files...")
        for p in chunk_files:
            try:
                os.remove(p)
            except OSError as e:
                printc(f"  <red>Error removing <i>{p}</i>: {e}</red>")
        printc("  <green>Cleanup complete.</green>")

def tokenize_and_create_chunks(enc):
    """
    Loads the dataset, filters, tokenizes, and saves data in chunks.
    """
    processed_docs_count, chunk_index = load_progress()
    eot_token_id = enc.encode_single_token(EOT_TOKEN)

    printc("<b>Loading dataset (streaming)...</b>")
    ds = load_dataset(hf_dataset_name, name=hf_dataset_config, split=hf_dataset_split, streaming=True)

    if processed_docs_count > 0:
        printc(f"<cyan>Resume:</cyan> skipping <b>{processed_docs_count:,}</b> already processed docs.")
        ds = ds.skip(processed_docs_count)
        chunk_index += 1

    val_docs_count = int(num_documents * val_split)
    train_docs_count = num_documents - val_docs_count

    train_chunk, val_chunk = [], []
    pbar = tqdm(initial=processed_docs_count, total=num_documents, unit="docs", desc="Processing documents")
    try:
        for doc in ds:
            if doc.get("language") == language_filter and doc.get("language_score", 0) >= language_score_threshold:
                text = doc.get("text")
                if text:
                    tokens = enc.encode(text, allowed_special="all")
                    tokens.append(eot_token_id)

                    if pbar.n < train_docs_count:
                        train_chunk.append(tokens)
                    else:
                        val_chunk.append(tokens)

            if len(train_chunk) >= CHUNK_SIZE or len(val_chunk) >= CHUNK_SIZE:
                if train_chunk:
                    save_chunk(train_chunk, chunk_index, "train")
                if val_chunk:
                    save_chunk(val_chunk, chunk_index, "val")

                processed_docs_count = pbar.n + 1
                save_progress(processed_docs_count, chunk_index)
                train_chunk, val_chunk = [], []
                chunk_index += 1

            pbar.update(1)

    except (KeyboardInterrupt, Exception) as e:
        printc(f"<br/><yellow>Process interrupted: {e}. Progress saved.</yellow>")
    finally:
        pbar.close()

    if train_chunk or val_chunk:
        if train_chunk: save_chunk(train_chunk, chunk_index, "train")
        if val_chunk: save_chunk(val_chunk, chunk_index, "val")

    if os.path.exists(state_file):
        os.remove(state_file)

    printc("<br/><b>Tokenization finished.</b>")
    return True

def main():
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "train.npz")) and os.path.exists(os.path.join(output_dir, "val.npz")):
        printc("<yellow>Final .npz files already exist. Skipping processing.</yellow>")
        return

    enc = load_tokenizer()
    tokenize_and_create_chunks(enc)

    printc("<br/><b>Starting consolidation process...</b>")
    consolidate_chunks_to_npz(temp_dir, output_dir, "train")
    consolidate_chunks_to_npz(temp_dir, output_dir, "val")

    try:
        if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        printc(f"<yellow>Could not remove temp directory: {e}</yellow>")

    printc("<br/><b><green>✅ All done.</green></b>")

if __name__ == "__main__":
    main()
