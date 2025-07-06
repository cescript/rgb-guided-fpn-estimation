import torch
import torch.nn as nn
import torch.nn.functional as F

# cross model block-wise attention module
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, block_size=8):
        super().__init__()
        # linear projection layers for Q, K, V
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # multi-head attention for small blocks
        num_heads = 4
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # residual normalization layer
        self.norm = nn.LayerNorm(embed_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.block_size = block_size

    def forward(self, ir_feat, rgb_feat):
        B, C, H, W = ir_feat.size()

        # H and W must be divisible by block_size
        assert H % self.block_size == 0 and W % self.block_size == 0, "H and W must be divisible by block_size"
        
        # project to Q, K, V over channel dimensions
        Q = self.q_proj(ir_feat)
        K = self.k_proj(rgb_feat)
        V = self.v_proj(rgb_feat)

        # split into non-overlapping blocks
        bs = self.block_size
        Q_blocks = Q.unfold(2, bs, bs).unfold(3, bs, bs)  # [B, C, nH, nW, bs, bs]
        K_blocks = K.unfold(2, bs, bs).unfold(3, bs, bs)
        V_blocks = V.unfold(2, bs, bs).unfold(3, bs, bs)

        # get the number of windows
        nH, nW = Q_blocks.shape[2], Q_blocks.shape[3]

        # reshape blocks to (B * nH * nW, bs*bs, C)
        Q_blocks = Q_blocks.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, bs * bs).transpose(1, 2)
        K_blocks = K_blocks.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, bs * bs).transpose(1, 2)
        V_blocks = V_blocks.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, bs * bs).transpose(1, 2)

        # apply attention per block [B*nH*nW, bs*bs, C]
        attn_out, _ = self.attn(Q_blocks, K_blocks, V_blocks)

        # add residual connection and normalize per block
        attn_out = self.norm(attn_out + Q_blocks)

        # reshape back to (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).reshape(B, nH, nW, C, bs, bs)
        attn_out = attn_out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return self.gamma * attn_out + ir_feat
