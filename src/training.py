import torch
import torch.nn.functional as F

# from DNA diffusion people code, and lucid rains.
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.detach()))

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(param.data)


def multi_task_loss(sequence_logits, x_0, predicted_scores, target_scores, mask, weight=0.5):
    sequence_logits = sequence_logits.view(-1, sequence_logits.size(-1))
    targets = x_0.argmax(dim=-1).view(-1)
    mask = mask.view(-1)
    
    valid_logits = sequence_logits[mask.bool()]
    valid_targets = targets[mask.bool()]

    reconstruction_loss = F.cross_entropy(valid_logits, valid_targets)
    score_loss = F.mse_loss(predicted_scores.squeeze(), target_scores)

    # Scale second loss by a ratio (or simply reconstruction_loss + weight * score_loss)
    total_loss = reconstruction_loss + (score_loss / score_loss.detach()) * weight
    return total_loss


def train_multi_task(model, diffusion, dataloader, optimizer, epochs, device, weight=1.0):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0, target_scores, mask = batch
            x_0 = x_0.to(device)
            target_scores = target_scores.to(device)
            mask = mask.to(device).bool()

            optimizer.zero_grad()
            sequence_logits, predicted_scores = diffusion(x_0, target_scores, mask)
            loss = multi_task_loss(sequence_logits, x_0, predicted_scores, target_scores, mask=mask, weight=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch+1}, loss: {avg_loss:.4f}")


def train(model, diffusion, dataloader, optimizer, epochs, device, ema_decay=0.995):
    ema = EMA(model, decay=ema_decay) 
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_0, scores, mask = batch
            x_0 = x_0.to(device)
            scores = scores.to(device)
            mask = mask.to(device).bool()

            optimizer.zero_grad()
            loss = diffusion(x_0, scores, mask)
            loss.backward()
            optimizer.step()

            ema.update(model)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    ema.apply_shadow(model)

