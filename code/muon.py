"""A compact Muon implementation based on the course's Newton-Schulz update."""
import math
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(gradient, steps=5):
    x = gradient.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        correction = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + correction @ x
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    """Momentum followed by Newton-Schulz orthogonalization for hidden matrices."""

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, decay, beta = group['lr'], group['weight_decay'], group['momentum']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(gradient)
                momentum = state['momentum_buffer']
                momentum.lerp_(gradient, 1 - beta)
                update = torch.lerp(gradient, momentum, beta)
                update = zeropower_via_newtonschulz5(update)
                update.mul_(math.sqrt(max(1., update.size(-2) / update.size(-1))))
                if decay:
                    parameter.mul_(1 - lr * decay)
                parameter.add_(update.to(parameter.dtype), alpha=-lr)
