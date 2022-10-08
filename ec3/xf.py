"""
from: 
https://raw.githubusercontent.com/openai/glide-text2im/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/xf.py
in turn: 
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
see also: 
https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/text2im_model.py
"""

import math

import torch as th
import torch.nn as nn


def convert_module_to_f16(l):
	"""
	Convert primitive modules to float16.
	"""
	if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
		l.weight.data = l.weight.data.half()
		if l.bias is not None:
			l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
	"""
	Implementation that supports fp16 inputs but fp32 gains/biases.
	"""

	def forward(self, x: th.Tensor):
		return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module): # this is only self-attention. 
	def __init__(self, n_ctx, width, heads):
		super().__init__()
		self.n_ctx = n_ctx
		self.width = width
		self.heads = heads
		self.c_qkv = nn.Linear(width, width * 3) # note! * 3
		self.c_proj = nn.Linear(width, width)
		self.attention = QKVMultiheadAttention(heads, n_ctx)

	def forward(self, x):
		x = self.c_qkv(x)
		x = self.attention(x)
		x = self.c_proj(x)
		return x


class MLP(nn.Module):
	def __init__(self, width):
		super().__init__()
		self.width = width
		self.c_fc = nn.Linear(width, width * 4) # ! expand by 4 ! 
		self.c_proj = nn.Linear(width * 4, width)
		self.gelu = nn.GELU()

	def forward(self, x):
		return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
	def __init__(self, n_heads: int, n_ctx: int):
		super().__init__()
		self.n_heads = n_heads
		self.n_ctx = n_ctx

	def forward(self, qkv):
		bs, n_ctx, width = qkv.shape # bs = batch size
		attn_ch = width // self.n_heads // 3 # input is expanded by 3, c_qkv
		scale = 1 / math.sqrt(math.sqrt(attn_ch))
		qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
		q, k, v = th.split(qkv, attn_ch, dim=-1)
		weight = th.einsum(
			"bthc,bshc->bhts", q * scale, k * scale
		)  # More stable with f16 than dividing afterwards
		wdtype = weight.dtype
		weight = th.softmax(weight.float(), dim=-1).type(wdtype)
		return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
	def __init__(
		self,
		n_ctx: int,
		width: int,
		heads: int,
	):
		super().__init__()

		self.attn = MultiheadAttention(
			n_ctx,
			width,
			heads,
		)
		self.ln_1 = LayerNorm(width)
		self.mlp = MLP(width)
		self.ln_2 = LayerNorm(width)

	def forward(self, x: th.Tensor):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class Transformer(nn.Module):
	def __init__(
		self,
		n_ctx: int, # number of text tokens to expect.
		width: int, # e.g. 512, width of the transformer
		layers: int, # depth
		heads: int, # eg 8, so each head has a width of 64. 
	):
		super().__init__()
		self.n_ctx = n_ctx
		self.width = width
		self.layers = layers
		self.resblocks = nn.ModuleList(
			[
					ResidualAttentionBlock(
						n_ctx, #tumber of tokens of context, fixed. 
						width,
						heads,
					)
					for _ in range(layers)
			]
		)

	def forward(self, x: th.Tensor):
		for block in self.resblocks:
			x = block(x)
		return x
