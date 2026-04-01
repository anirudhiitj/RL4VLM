"""
Shared Adam optimizer for A3C.

Uses shared memory for optimizer state (moving averages) so that
asynchronous workers can push gradients to a shared global model.
"""
import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    """
    Adam optimizer with shared state across processes.
    All state tensors are placed in shared memory so multiple
    torch.multiprocessing workers can access them.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)
        # Move state to shared memory
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Share in memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
