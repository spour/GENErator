import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6, kernel_size=5, dilation=1):
        super(ConvEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2 * dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            input_dim = hidden_dim  
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv(x)
        x = x.permute(0, 2, 1)  
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ).cuda()
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        t_emb = self.mlp(emb)
        return t_emb


class ScoreEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(ScoreEmbedder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.shift = nn.Parameter(torch.zeros(hidden_size))
        self.scale = nn.Parameter(torch.ones(hidden_size))
      
        self.activation = nn.GELU()

    def forward(self, scores):
        x = self.activation(self.linear1(scores))
        x = self.activation(self.linear2(x))
        x = self.layer_norm(x)
        x = x * self.scale + self.shift
        return x


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_relative_position=16):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.max_relative_position = max_relative_position

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * max_relative_position + 1, num_heads))
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # AATTENNTION
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        relative_positions = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        relative_positions = relative_positions.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position

        relative_bias = self.relative_position_bias[relative_positions]
        relative_bias = relative_bias.permute(2, 0, 1)
        attn_scores = attn_scores + relative_bias.unsqueeze(0)

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        return attn_output


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, stochastic_depth_p=0.1):
        super(DiffusionTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        # layer is active 90% of time with stoch depth = 0.1
        self.stochastic_depth_p = stochastic_depth_p
        
    def forward(self, x, attn_mask=None):
      # stochastic depth mechanismto improve training for deep neural networks by 
       # randomly skipping layers during forward passes.
      

        if self.training:
            survival_prob = 1.0 - self.stochastic_depth_p
            batch_size = x.size(1)
             # SHOULD I SKIP?
            random_tensor = torch.rand(1, batch_size, 1, device=x.device) + survival_prob
            binary_mask = torch.floor(random_tensor)
            x = x * binary_mask
        else:
            survival_prob = 1.0

        x_residual = x
        x2, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = x_residual + self.dropout1(x2)
        x = self.norm1(x)

        x_residual = x
        x2 = self.linear2(self.activation(self.linear1(x)))
        x = x_residual + self.dropout2(x2)
        x = self.norm2(x)

        if self.training:
            x = x / survival_prob

        return x


class DiffusionModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers=6, num_heads=8, self_conditioning=True, max_seq_len=512):
        super(DiffusionModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.self_conditioning = self_conditioning

        input_size = num_classes * 2 if self_conditioning else num_classes

        self.conv_encoder = ConvEncoder(input_dim=input_size, hidden_dim=hidden_size).cuda()
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.score_embedder = ScoreEmbedder(1, hidden_size)

        self.positional_encoding = nn.Parameter(self.get_sinusoid_encoding_table(max_seq_len, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads).cuda() for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_size, num_classes).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_sinusoid_encoding_table(self, max_seq_len, hidden_size):
        def get_angle(pos, i_hidden):
            return pos / (10000 ** (2 * (i_hidden // 2) / hidden_size))
        sinusoid_table = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i_hidden in range(hidden_size):
                angle = get_angle(pos, i_hidden)
                if i_hidden % 2 == 0:
                    sinusoid_table[pos, i_hidden] = math.sin(angle)
                else:
                    sinusoid_table[pos, i_hidden] = math.cos(angle)
        return sinusoid_table.unsqueeze(0)

    def forward(self, x_t, t, score, x_self_cond=None, attention_mask=None):
        x_t.requires_grad_(True)
        if self.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x_t)
            x_input = torch.cat([x_t, x_self_cond], dim=-1) # concatenate for self conditioning
        else:
            x_input = x_t

        x = self.conv_encoder(x_input)
        t_emb = self.timestep_embedder(t)

        if score.dim() == 1:
            score = score.unsqueeze(-1)
        elif score.shape[1] != 1:
            score = score[:, :1]

        score_emb = self.score_embedder(score)

        seq_len = x.size(1)
        conditioning = t_emb + score_emb
        conditioning = conditioning.unsqueeze(1)
        x = x + conditioning.expand(-1, seq_len, -1) # combine and broadcast the score and time emb

        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc

        x = x.transpose(0, 1)

        if attention_mask is not None:
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.squeeze(-1)
            attention_mask = attention_mask.bool()
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        for block in self.blocks:
            x = block(x, attn_mask=key_padding_mask)

        x = x.transpose(0, 1)
        logits = self.output_layer(x)
        return logits


class MultiTaskDiffusionModel(DiffusionModel):
  # MODEL DOES BETTER WITH THE SCORE HEAD TOO, NOT JUST DIFFUSION 
    def __init__(self, num_classes, hidden_size, **kwargs):
        super().__init__(num_classes, hidden_size, **kwargs)
        self.score_prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).cuda()

    def forward(self, x_t, t, score, x_self_cond=None, attention_mask=None):
        if self.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x_t)
            x_input = torch.cat([x_t, x_self_cond], dim=-1)
        else:
            x_input = x_t

        x = self.conv_encoder(x_input)
        t_emb = self.timestep_embedder(t)
        score_emb = self.score_embedder(score.unsqueeze(-1) if score.dim() == 1 else score)

        seq_len = x.size(1)
        conditioning = t_emb + score_emb
        conditioning = conditioning.unsqueeze(1)
        x = x + conditioning.expand(-1, seq_len, -1)

        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc

        x = x.transpose(0, 1)

        if attention_mask is not None:
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.squeeze(-1)
            attention_mask = attention_mask.bool()
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None

        for block in self.blocks:
            x = block(x, attn_mask=key_padding_mask)

        x = x.transpose(0, 1)
        sequence_logits = self.output_layer(x)
        x_hidden = x.mean(dim=1)
        predicted_scores = self.score_prediction_head(x_hidden)
        return sequence_logits, predicted_scores
