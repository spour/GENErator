import torch
import math
import torch.nn.functional as F

# a lot is from lucid rains code, and DNAdiffusion pinello. 
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, min=0.0001, max=0.02)

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, s=0.008, type='linear'):
    if type == 'linear':
        return torch.linspace(beta_start, beta_end, T)
    elif type == 'cosine':
        return cosine_beta_schedule(T, s=s)

def extract(a, t, x_shape, device=None):
    if device:
        a = a.to(device)
        t = t.to(device)
    out = a.gather(-1, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))
