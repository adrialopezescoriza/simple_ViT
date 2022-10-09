import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w)
    patch_size = h // n_patches

    # TODO: Make efficient
    for idx, image in enumerate(images):
        for i in range(patches):
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


class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d=8):
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

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

    def forward(self, images):
        patches = patchify(images, self.n_patches)
        # Build vector representations
        tokens = self.linear_mapper(patches)

        # Stack classification tokens to tokens
        tokens = torch.stack([torch.vstack(self.class_token, tokens[i]) for i in range(len(tokens))])
        return tokens
