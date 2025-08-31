import torch
from torch.optim.optimizer import Optimizer

import triton
import triton.language as tl

@triton.jit
def _update_hessian_kernel(
    hessian_ptr,
    grad_ptr,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    hessian = tl.load(hessian_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    hessian_fp32 = hessian.to(tl.float32)
    grad_fp32 = grad.to(tl.float32)

    new_hessian = beta2 * hessian_fp32 + (1.0 - beta2) * grad_fp32 * grad_fp32
    
    tl.store(hessian_ptr + offsets, new_hessian.to(hessian.dtype), mask=mask)


@triton.jit
def _step_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    hessian_ptr,
    lr,
    beta1,
    rho,
    bs,
    weight_decay,
    eps,
    n_elements,
    p_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    hessian = tl.load(hessian_ptr + offsets, mask=mask)

    p_fp32 = p.to(tl.float32)
    grad_fp32 = grad.to(tl.float32)
    exp_avg_fp32 = exp_avg.to(tl.float32)
    hessian_fp32 = hessian.to(tl.float32)

    new_exp_avg = beta1 * exp_avg_fp32 + (1.0 - beta1) * grad_fp32

    p_decayed = p_fp32 * (1.0 - lr * weight_decay)
    
    denominator = tl.maximum(rho * bs * hessian_fp32, eps)
    ratio = tl.abs(new_exp_avg) / denominator
    clamped_ratio = tl.minimum(ratio, 1.0)
    
    sign_new_exp_avg = tl.where(new_exp_avg > 0, 1.0, tl.where(new_exp_avg < 0, -1.0, 0.0))
    update = lr * sign_new_exp_avg * clamped_ratio
    new_p = p_decayed - update

    tl.store(p_ptr + offsets, new_p.to(p_dtype), mask=mask)
    tl.store(exp_avg_ptr + offsets, new_exp_avg.to(exp_avg.dtype), mask=mask)


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False, eps: float = 1e-15, bs: int):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if capturable:
            raise ValueError("Capturable mode is not supported by this Triton implementation.")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not bs > 0:
            raise ValueError(f"Invalid batch size (bs): {bs}")
        
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        maximize=maximize, eps=eps, bs=bs)
        super(SophiaG, self).__init__(params, defaults)
        
        self.hessian_update_stream = torch.cuda.Stream()

    def _init_state(self, p):
        """Initializes optimizer state for a parameter."""
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def update_hessian(self):
        """
        Synchronizes the Hessian update stream with the current stream.
        This ensures that the Hessian update from the previous step is complete
        before the current optimizer step uses it. Also handles state initialization.
        """
        torch.cuda.current_stream().wait_stream(self.hessian_update_stream)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                
                if len(state) > 0 and state['exp_avg'].shape != p.shape:
                    print(f"SophiaG: Detected shape mismatch for a parameter (state: {state['exp_avg'].shape}, param: {p.shape}). Re-initializing state.")
                    state.clear()
                    
                self._init_state(p)

    @torch.no_grad()
    def schedule_hessian_update(self):
        """
        This allows the update to overlap with the backward pass of the next iteration,
        hiding its latency and improving GPU utilization.
        """
        with torch.cuda.stream(self.hessian_update_stream):
            for group in self.param_groups:
                beta1, beta2 = group['betas']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('SophiaG does not support sparse gradients')

                    state = self.state[p]
                    
                    if len(state) == 0:
                        raise RuntimeError(f"SophiaG: State not initialized for parameter with shape {p.shape}, but it has a gradient. Ensure `optimizer.update_hessian()` is called before `backward()`.")

                    hessian = state['hessian']
                    n_elements = p.numel()
                    
                    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                    _update_hessian_kernel[grid](
                        hessian, grad, beta2, n_elements, BLOCK_SIZE=1024
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            rho = group['rho']
            eps = group['eps']
            bs = group['bs']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if maximize:
                    grad = -grad
                
                if grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    raise RuntimeError("Optimizer state not initialized. Call update_hessian() before step().")

                state['step'] += 1
                
                exp_avg = state['exp_avg']
                hessian = state['hessian']
                n_elements = p.numel()

                p_dtype = p.dtype
                if p_dtype == torch.float16:
                    p_dtype_tl = tl.float16
                elif p_dtype == torch.bfloat16:
                    p_dtype_tl = tl.bfloat16
                else:
                    p_dtype_tl = tl.float32

                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                _step_kernel[grid](
                    p, grad, exp_avg, hessian,
                    lr, beta1, rho, float(bs), weight_decay, eps,
                    n_elements,
                    p_dtype=p_dtype_tl,
                    BLOCK_SIZE=1024,
                )
        return loss
