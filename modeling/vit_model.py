import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w // n_patches ** 2)
    patch_size = h // n_patches

    # TODO: Make efficient (parallelize)
    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()

    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d)) if j % 2 == 0 else np.cos(i / (10000 ** ((j * i) / d))))
    return result

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        self.d = d
        self.n_heads = n_heads
        super(MyMSA, self).__init__()

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        # Construct MLP for each of the self-attention elements
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape   (N, seq_length, token_dim)
        # We go into shape      (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to      (N, seq_length, item_dim) (through concatenation)

        result = []
        for sequence in sequences:
            seq_result = []
            for head_idx in range(self.n_heads):
                q_mapping = self.q_mappings[head_idx]
                k_mapping = self.k_mappings[head_idx]
                v_mapping = self.v_mappings[head_idx]

                # Extract relevant part of the sequence for the given head
                seq = sequence[:, head_idx * self.d_head: (head_idx + 1) * self.d_head]

                # Compute query, hey, value
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                # Attention formula from paper
                attention = self.softmax((q @ k.T) / (self.d_head ** 0.5))

                # Extract and append result
                seq_result.append(attention @ v)

            # Append result
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])




class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_d)
        self.norm2 = nn.LayerNorm(hidden_d)

        self.mhsa = MyMSA(hidden_d, n_heads)
        self.mlp = nn.Sequential(
                    nn.Linear(hidden_d, mlp_ratio * hidden_d),
                    nn.GELU(),
                    nn.Linear(mlp_ratio * hidden_d, hidden_d))

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = self.norm2(out) + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw=(1, 32, 32), n_patches=8, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        # Shape assertions
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number or patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token (random initialization)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embeddings (not learnable)
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d).clone().detach())
        self.pos_embed.requires_grad = False

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLP
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=1))

    def forward(self, images):
        # Images dimensions
        n, c, h, w = images.shape

        # Patchify
        patches = patchify(images, self.n_patches).to(images.device)

        # Linear mapping: build vector representations
        tokens = self.linear_mapper(patches)

        # Stack classification tokens to tokens
        tokens = torch.stack([torch.vstack([self.class_token, tokens[i]]) for i in range(len(tokens))])

        # Add positional embeddings (Vaswani et al.: https://arxiv.org/abs/1706.03762)
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens * pos_embed

        # Encoding through transformer blocks
        for block in self.blocks:
            out = block(out)

        # Getting classification token only
        out = out[:, 0]

        # Map to output dimension, output category distribution
        return self.mlp(out)
