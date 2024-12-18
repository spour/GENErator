import torch
import torch.nn.functional as F
from .utils import default, get_beta_schedule, extract

class DiscreteDiffusion:
  # a lot is from the lucid rains code, and the DNA diffusion people 
    def __init__(self, model, timesteps, num_classes=4, self_conditioning=True, mode="single"):
        self.model = model.cuda()
        self.num_classes = num_classes
        self.T = timesteps
        self.betas = get_beta_schedule(timesteps).cuda()
        self.timesteps = timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.betas.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.self_conditioning = self_conditioning
        self.mode = mode

        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, scores, shape, cond_weight):
        return self.p_sample_loop(scores=scores, shape=shape, cond_weight=cond_weight)

    @torch.no_grad()
    def p_sample_loop(self, scores, shape, cond_weight):
        device = self.device
        img = torch.randn(shape, device=device)
        imgs = []
        x_self_cond = None

        batch_size = shape[0]
        context_mask = torch.ones(batch_size, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample_guided(
                x=img,
                t=t,
                t_index=i,
                scores=scores,
                cond_weight=cond_weight,
                context_mask=context_mask,
                x_self_cond=x_self_cond
            )
            imgs.append(img.cpu())
            x_self_cond = img.detach()

        return imgs

    @torch.no_grad()
    def p_sample_guided(self, x, scores, t, t_index, context_mask, cond_weight, x_self_cond=None):
      # do  one reverse diffusion step with cfg.
        batch_size = x.shape[0]
        device = self.device

        scores = scores.view(batch_size, -1)
        t = t.view(batch_size)
        context_mask = context_mask.view(batch_size)

        x_double = torch.cat([x, x], dim=0) # double up for conditional and unconditional passes
        t_double = torch.cat([t, t], dim=0) 
        scores_double = torch.cat([scores, scores], dim=0)
        context_mask_double = torch.cat([context_mask, torch.zeros_like(context_mask)], dim=0)

        if x_self_cond is not None:
            x_self_cond_double = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            x_self_cond_double = None
        # eps cond is noise predicted under conditioning, eps_uncond is without
        if self.mode == "single":
            predicted_noise = self.model(x_double, t_double, scores_double, x_self_cond=x_self_cond_double, attention_mask=None)
            eps_cond, eps_uncond = predicted_noise.chunk(2, dim=0)
        elif self.mode == "multi":
            predicted_noise, _ = self.model(x_double, t_double, scores_double, x_self_cond=x_self_cond_double, attention_mask=None)
            eps_cond, eps_uncond = predicted_noise.chunk(2, dim=0)

        x_t = eps_uncond + cond_weight * (eps_cond - eps_uncond) #  add scaled difference to improve conditioning (cfg)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * x_t / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, scores, mask=None, noise=None, loss_type="huber", p_uncond=0.1):
        noise = default(noise, torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        context_mask = torch.bernoulli(torch.ones(scores.shape[0], device=self.device) * (1 - p_uncond))

        scores_masked = scores.clone()
        scores_masked[context_mask == 0] = 0
        if scores_masked.dim() == 1:
            scores_masked = scores_masked.unsqueeze(-1)

        if self.mode == "single":
            predicted_noise = self.model(x_noisy, t, scores_masked, attention_mask=mask)
            if loss_type == "l1":
                loss = F.l1_loss(noise, predicted_noise)
            elif loss_type == "l2":
                loss = F.mse_loss(noise, predicted_noise)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)
            else:
                raise NotImplementedError()
            return loss
        elif self.mode == "multi":
            sequence_logits, predicted_scores = self.model(x_noisy, t, scores_masked, attention_mask=mask)
            if loss_type == "l1":
                sequence_loss = F.l1_loss(noise, sequence_logits)
            elif loss_type == "l2":
                sequence_loss = F.mse_loss(noise, sequence_logits)
            elif loss_type == "huber":
                sequence_loss = F.smooth_l1_loss(noise, sequence_logits)
            else:
                raise NotImplementedError()

            score_loss = F.mse_loss(predicted_scores.squeeze(), scores)
            return sequence_loss + score_loss

    def forward(self, x, scores, mask=None):
        x = x.float()
        scores = scores.float().unsqueeze(-1) if scores.dim() == 1 else scores.float()
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()

        if self.mode == "single":
            return self.p_losses(x, t, scores, mask=mask)
        else:
            sequence_logits, predicted_scores = self.model(x, t, scores, attention_mask=mask)
            return sequence_logits, predicted_scores
