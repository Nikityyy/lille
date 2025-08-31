import os
import time
import math
import pickle
import threading
import collections
import queue
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import tiktoken
import wandb
from html2term import printc

from sophia_triton import SophiaG
from model import GPT, GPTConfig

# --- General Settings ---
out_dir = 'checkpoints'
eval_interval = 500
log_interval = 1
eval_iters = 100
# resume_checkpoint = None
resume_checkpoint = "checkpoints_ft/ckpt_iter_73000.pt"

# --- Finetuning Settings ---
finetune = True
finetune_out_dir = 'checkpoints_ft'
finetune_data_dir = 'data/smol-sft'
finetune_learning_rate = 1e-5
finetune_num_epochs = 3

# --- W&B Logging ---
wandb_log = True
wandb_project = 'modern-gpt-pretrain'
wandb_run_name = f'run-modern-gpt-{time.strftime("%Y-%m-%d-%H-%M-%S")}'

# --- Data Settings ---
data_dir = 'data/fineweb_edu_sample_10BT'
pretrain_data_dir = data_dir
batch_size = 16
block_size = 512
num_epochs = 1
gradient_accumulation_steps = 2

# --- Model Architecture ---
n_layers = 24
n_embd = 640
n_heads = 10
n_kv_heads = 2
dropout = 0.1
layer_norm_eps = 1e-5

# --- Optimizer & LR Schedule ---
learning_rate = 1e-4
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
min_lr = learning_rate / 10
hess_interval = 10

# --- Hardware & Performance ---
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

class NpzDataset:
    """A simple lazy-loading dataset for the tokens/offsets .npz format."""
    def __init__(self, file_path):
        self.data = np.load(file_path, mmap_mode='r')
        self.tokens = self.data['tokens']
        self.offsets = self.data['offsets']
        self.num_docs = len(self.offsets) - 1

    def __len__(self):
        return self.num_docs

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        return self.tokens[start:end].tolist()

class DataPrefetcher:
    """ An asynchronous data prefetcher that prepares batches on the CPU. """
    def __init__(self, data, block_size, batch_size, max_prefetch=2):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.is_running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def _preload(self):
        return get_batch('train')

    def run(self):
        while self.is_running:
            try:
                self.queue.put(self._preload(), timeout=1)
            except queue.Full:
                continue

    def next(self):
        return self.queue.get()

    def close(self):
        self.is_running = False
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join()

def get_batch(split, pretrain=False):
    """
    Get a batch of data. Handles padding for sequences shorter than block_size.
    For supervised fine-tuning, it masks out the loss for prompt tokens.
    """
    if pretrain:
        data = train_data_pretrain if split == 'train' else val_data_pretrain
    else:
        data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data), (batch_size,))
    batch_raw = [data[i] for i in ix]

    x_padded = torch.full((batch_size, block_size), pad_token_id, dtype=torch.long)
    y_padded = torch.full((batch_size, block_size), -100, dtype=torch.long)

    is_finetune_split = finetune and not pretrain

    for i, tokens in enumerate(batch_raw):
        seq_len = min(len(tokens), block_size)
        x_padded[i, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
        
        targets = torch.tensor(tokens[1:seq_len], dtype=torch.long)
        
        if is_finetune_split and assistant_token_id is not None:
            x_seq = x_padded[i, :seq_len]
            assistant_indices = (x_seq == assistant_token_id).nonzero(as_tuple=True)[0]
            
            if len(assistant_indices) > 0:
                last_assistant_idx = assistant_indices[-1]
                targets[:last_assistant_idx] = -100

        y_padded[i, :seq_len-1] = targets

    return x_padded, y_padded

@torch.no_grad()
def estimate_loss(pretrain=False):
    """
    Estimate loss on train and validation splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X_cpu, Y_cpu = get_batch(split, pretrain=pretrain)
            X = X_cpu.to(device, non_blocking=True)
            Y = Y_cpu.to(device, non_blocking=True)
            attn_mask = (X != pad_token_id)
            with ctx:
                logits, _ = model(X, attn_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss
        if ddp:
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= ddp_world_size
        out[split] = losses.mean()
    model.train()
    return out

def configure_optimizers(model, weight_decay, learning_rate, betas):
    """
    Configure optimizer with weight decay for 2D parameters.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        printc(f"  <#cccccc>Num decayed parameter tensors:</#cccccc> <b>{len(decay_params)}</b>, with <b>{num_decay_params:,}</b> parameters")
        printc(f"  <#cccccc>Num non-decayed parameter tensors:</#cccccc> <b>{len(nodecay_params)}</b>, with <b>{num_nodecay_params:,}</b> parameters<br/>")
    optimizer = SophiaG(optim_groups, lr=learning_rate, betas=betas, rho=0.05, weight_decay=weight_decay, eps=1e-15, bs=tokens_per_optimizer_step)
    return optimizer

def get_cosine_schedule_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio_val=0.1):
    """
    Create a learning rate scheduler with a cosine decay and linear warmup.
    """
    def lr_lambda_func(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio_val + (1.0 - min_lr_ratio_val) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_func)

def save_checkpoint_async(checkpoint, path, force=False, old_path_to_delete=None):
    """
    Save a checkpoint asynchronously in a separate thread.
    """
    temp_path = path + ".tmp"
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
        if old_path_to_delete and os.path.exists(old_path_to_delete):
            os.remove(old_path_to_delete)
    except Exception as e:
        printc(f"<red>Error saving checkpoint to {path}: {e}</red>")
        if os.path.exists(temp_path):
            os.remove(temp_path)

if finetune:
    out_dir = finetune_out_dir
    learning_rate = finetune_learning_rate
    num_epochs = finetune_num_epochs
    data_dir = finetune_data_dir
    warmup_iters = 1000
    weight_decay = 0.01
    dropout = 0.0
    wandb_project = 'modern-gpt-finetune'
    if not resume_checkpoint:
        raise ValueError("For finetuning, a `resume_checkpoint` must be provided.")
    printc("<blue>" + "="*50 + "</blue>")
    printc("<b><cyan>|| FINETUNING MODE ENABLED</cyan></b>")
    printc(f"|| Output directory: <i>{out_dir}</i>")
    printc(f"|| Data directory: <i>{data_dir}</i>")
    printc(f"|| Learning rate: <b>{learning_rate}</b>")
    printc(f"|| Epochs: <b>{num_epochs}</b>")
    printc("<blue>" + "="*50 + "</blue><br/>")
else:
    printc("<blue>" + "="*50 + "</blue>")
    printc("<b><cyan>|| PRETRAINING MODE ENABLED</cyan></b>")
    printc(f"|| Output directory: <i>{out_dir}</i>")
    printc(f"|| Data directory: <i>{data_dir}</i>")
    printc(f"|| Learning rate: <b>{learning_rate}</b>")
    printc(f"|| Epochs: <b>{num_epochs}</b>")
    printc("<blue>" + "="*50 + "</blue><br/>")

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type, dtype=ptdtype)

if wandb_log and master_process:
    config_dict = {k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool))}
    wandb.init(project=wandb_project, name=wandb_run_name, config=config_dict)

with open('tokenizer/Hastings.pkl', 'rb') as f:
    hastings = pickle.load(f)
enc = tiktoken.core.Encoding(hastings.pop('name'), **hastings)
vocab_size = enc.n_vocab
assistant_token_id = enc.encode_single_token("<|assistant|>")

try:
    pad_token_id = enc.encode_single_token("<|pad|>")
except KeyError:
    printc("<yellow>Warning:</yellow> '<b>&lt;|pad|&gt;</b>' token not found. Using '<b>&lt;|endoftext|&gt;</b>' as a pad token.")
    pad_token_id = enc.encode_single_token("<|endoftext|>")

printc("<br/><b>Loading dataset using NpzDataset...</b>")
train_data = NpzDataset(os.path.join(data_dir, 'train.npz'))
val_data = NpzDataset(os.path.join(data_dir, 'val.npz'))
if data_dir != pretrain_data_dir:
    train_data_pretrain = NpzDataset(os.path.join(pretrain_data_dir, 'train.npz'))
    val_data_pretrain = NpzDataset(os.path.join(pretrain_data_dir, 'val.npz'))
else:
    train_data_pretrain, val_data_pretrain = train_data, val_data

train_tokens = len(train_data.tokens)
tokens_per_optimizer_step = batch_size * block_size * ddp_world_size * gradient_accumulation_steps
max_optimizer_steps = (train_tokens // tokens_per_optimizer_step) * num_epochs
iters_per_epoch_optimizer_steps = train_tokens // tokens_per_optimizer_step
lr_decay_iters = max_optimizer_steps

model_args = dict(
    n_layers=n_layers, n_embd=n_embd, vocab_size=vocab_size, block_size=block_size,
    dropout=dropout, n_heads=n_heads, n_kv_heads=n_kv_heads, layer_norm_eps=layer_norm_eps
)
config = GPTConfig(**model_args)
model = GPT(config)
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
if master_process:
    printc(f"Model has <b>{num_params / 1e6:.2f}M</b> parameters.")

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2))

min_lr_ratio_for_scheduler = min_lr / learning_rate
scheduler = get_cosine_schedule_with_warmup_scheduler(
    optimizer, num_warmup_steps=warmup_iters,
    num_training_steps=max_optimizer_steps, min_lr_ratio_val=min_lr_ratio_for_scheduler
)

iter_num = 0
best_val_loss = 1e9
if resume_checkpoint and os.path.exists(resume_checkpoint):
    if master_process: printc(f"<b>Loading checkpoint:</b> <i>{resume_checkpoint}</i>")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    ckpt_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        if k not in ckpt_model_args or ckpt_model_args[k] != v:
            if master_process: printc(f"  <yellow>Warning:</yellow> Mismatch in model config: '<b>{k}</b>'")
    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    if not finetune or (finetune and 'optimizer_state_dict' in checkpoint and resume_checkpoint.startswith(finetune_out_dir)):
        if master_process: printc("Resuming training with optimizer state.")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group.setdefault('bs', tokens_per_optimizer_step)
            group.setdefault('eps', 1e-15)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        scheduler.last_epoch = iter_num

if compile:
    if master_process: printc("<b>Compiling the model...</b>")
    model = torch.compile(model, backend="inductor", mode="max-autotune")
    if master_process:
        printc("  Warming up the compiled model...")
        with ctx:
            with torch.no_grad():
                x_warm_cpu, _ = get_batch('train')
                x_warm = x_warm_cpu.to(device, non_blocking=True)
                attn_mask_warm = (x_warm != pad_token_id)
                _, _ = model(x_warm, attn_mask=attn_mask_warm)
        printc("  <green>Warm-up complete.</green><br/>")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

if master_process: printc("<b>Setting up asynchronous data prefetcher for training...</b>")
train_prefetcher = DataPrefetcher(train_data, block_size, batch_size)

t0 = time.time()
checkpoint_threads = collections.deque()
last_interval_checkpoint_path = None
pbar = tqdm(range(iter_num, max_optimizer_steps), disable=not master_process)

for optimizer_step in pbar:
    if optimizer_step > iter_num and optimizer_step % eval_interval == 0:
        losses = estimate_loss()
        if finetune:
            losses_pt = estimate_loss(pretrain=True)
        current_epoch = optimizer_step / iters_per_epoch_optimizer_steps
        if master_process:
            printc(f"<br/><b>Epoch {current_epoch:.2f} | Step {optimizer_step}</b>")
            if finetune:
                printc(f"  <cyan>Finetune Loss</cyan> -> Train: <b>{losses['train']:.4f}</b>, Val: <b>{losses['val']:.4f}</b>")
                printc(f"  <#cccccc>Pretrain Loss</#cccccc> -> Train: <b>{losses_pt['train']:.4f}</b>, Val: <b>{losses_pt['val']:.4f}</b>")
            else:
                printc(f"  <cyan>Pretrain Loss</cyan> -> Train: <b>{losses['train']:.4f}</b>, Val: <b>{losses['val']:.4f}</b>")
            if wandb_log:
                log_data = {'eval/train_loss': losses['train'], 'eval/val_loss': losses['val'], 'trainer/epoch': current_epoch}
                if finetune:
                    log_data.update({'eval/pretrain_train_loss': losses_pt['train'], 'eval/pretrain_val_loss': losses_pt['val']})
                wandb.log(log_data, step=optimizer_step)

            while checkpoint_threads and not checkpoint_threads[0].is_alive():
                checkpoint_threads.popleft().join()
            raw_model = model.module if ddp else model
            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_args': raw_model.config.to_dict(),
                'iter_num': optimizer_step,
                'best_val_loss': best_val_loss
            }
            checkpoint_path = os.path.join(out_dir, f'ckpt_iter_{optimizer_step}.pt')
            thread = threading.Thread(target=save_checkpoint_async, args=(checkpoint.copy(), checkpoint_path, False, last_interval_checkpoint_path))
            thread.start()
            checkpoint_threads.append(thread)
            last_interval_checkpoint_path = checkpoint_path

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint['best_val_loss'] = best_val_loss
                best_model_path = os.path.join(out_dir, 'best_model.pt')
                thread = threading.Thread(target=save_checkpoint_async, args=(checkpoint.copy(), best_model_path, True))
                thread.start()
                checkpoint_threads.append(thread)
                printc(f"  <green>Started saving new best model to</green> <i>{best_model_path}</i>")

    optimizer.zero_grad(set_to_none=True)

    if compile and device_type == 'cuda':
        torch.compiler.cudagraph_mark_step_begin()

    for micro_step in range(gradient_accumulation_steps):
        X_cpu, Y_cpu = train_prefetcher.next()
        X = X_cpu.pin_memory().to(device, non_blocking=True)
        Y = Y_cpu.pin_memory().to(device, non_blocking=True)
        attn_mask = (X != pad_token_id)
        with ctx:
            logits, _ = model(X, attn_mask=attn_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) / gradient_accumulation_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
    if optimizer_step == 0:
        optimizer.update_hessian()
        
    scaler.step(optimizer)
    scaler.update()

    if optimizer_step % hess_interval == 0:
        with ctx:
            logits, _ = model(X, attn_mask=attn_mask)
        probs = F.softmax(logits, dim=-1)
        y_sample = torch.multinomial(probs.view(-1, logits.size(-1)), num_samples=1).view_as(Y)
        loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1))
        scaler.scale(loss_sampled).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.schedule_hessian_update()
        optimizer.zero_grad(set_to_none=True)

    if decay_lr:
        scheduler.step()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if optimizer_step % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"step {optimizer_step + 1}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {current_lr:e}")
        if wandb_log:
            wandb.log({'train/loss': lossf, 'trainer/lr': current_lr, 'trainer/dt_ms': dt * 1000}, step=optimizer_step)

pbar.close()
if master_process: printc("<br/><b>Training loop finished.</b> Closing data prefetcher...")
train_prefetcher.close()

if master_process:
    printc("<b>Saving final model and waiting for all saves to complete...</b>")
    raw_model = model.module if ddp else model
    final_checkpoint = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_args': raw_model.config.to_dict(),
        'iter_num': max_optimizer_steps,
        'best_val_loss': best_val_loss
    }
    final_checkpoint_path = os.path.join(out_dir, 'ckpt_final.pt')
    thread = threading.Thread(target=save_checkpoint_async, args=(final_checkpoint, final_checkpoint_path, True))
    thread.start()
    checkpoint_threads.append(thread)

    while checkpoint_threads:
        printc(f"  <#cccccc>Waiting for {len(checkpoint_threads)} remaining checkpoint(s) to save...</#cccccc>")
        checkpoint_threads.popleft().join()

    if wandb_log:
        wandb.finish()

if ddp:
    dist.destroy_process_group()

printc("<br/><b><green>✅ Training complete and all checkpoints saved.</green></b>")
