import torch as th
from torch import nn
import torch.cuda.amp
from ctypes import *
import clip_model
from pathlib import Path
from typing import Union
import torch._dynamo as dynamo
from constants import CHECKPOINTS_ROOT

dynamo.config.verbose=True
# note: I can't seem to get this to work. tlh April 7 2023



class Recognizer(nn.Module):
	CHECKPOINT_SAVEPATH = CHECKPOINTS_ROOT / "recognizer_checkpoint.ptx"
	
	def __init__(
		self,
		image_resolution: int, 
		vision_width:int, 
		patch_size:int,  
		prog_width:int, 
		embed_dim:int, 
		v_ctx:int, 
		p_ctx:int, 
		p_indim:int, 
		e_indim:int
		): 
		super().__init__()
		self.v_ctx = v_ctx
		self.p_ctx = p_ctx
		self.p_indim = p_indim
		self.prog_width = prog_width
		
		self.vit = clip_model.VisionTransformer(
			input_resolution = image_resolution, 
			patch_size = patch_size, 
			width = vision_width, 
			layers = 4, 
			heads = 8, 
			output_dim = embed_dim)

		self.vit_to_prt = nn.Linear(embed_dim, self.prog_width)
		
		self.encoder = nn.Linear(p_indim, prog_width)
		self.prt = clip_model.Transformer(
			width = prog_width, 
			layers = 6, 
			heads = 8, 
			attn_mask = self.build_attention_mask(v_ctx, p_ctx))
			
		self.prt_to_edit = nn.Linear(prog_width * (p_ctx + v_ctx), e_indim)
		self.ln_post = clip_model.LayerNorm(e_indim)
		self.gelu = clip_model.QuickGELU()
		self.tok_softmax = nn.Softmax(dim = 1)
	
	def forward(self, u, batch_a, batch_p): 
		# encode the image (we should only need to do this once??)
		q = th.zeros(6) # ! this will be parallelized !
		vx = self.vit(batch_a) # x is size [bs, v_ctx, 256] 
		q[0] = th.std(vx)
		vx = self.vit_to_prt(vx)
		q[1] = th.std(vx)
		# vx = gelu(vx) # ? needed ? 

		px = self.encoder(batch_p)
		q[2] = th.std(px)
		vxpx = th.cat((vx, px), dim = 1)
		q[3] = th.std(vxpx)
		# x = vxpx * mask
		x = self.prt(vxpx) # bs, v_ctx + p_ctx, prog_width
		q[4] = th.std(x)
		x = th.reshape(x, (-1,(self.v_ctx + self.p_ctx)*self.prog_width))
		# batch size will vary with dataparallel
		x = self.prt_to_edit(x)
		q[5] = th.std(x)
		# x = self.ln_post(x) # scale the inputs to softmax
		# x = self.gelu(x)
		# x = th.cat((self.tok_softmax(x[:,0:4]),
		# 		  self.tok_softmax(x[:,4:4+toklen]), 
		# 		  x[:,4+toklen:]), dim=1) -- this is for fourier position enc. 
		return x,q

	@staticmethod
	def build_attention_mask(v_ctx, p_ctx):
		# allow the model to attend to everything when predicting an edit
		# causalty is enforced by the editing process.
		# see ec31.py for a causal mask.
		ctx = v_ctx + p_ctx
		mask = th.ones(ctx, ctx)
		return mask
	
	def load_checkpoint(self, path: Union[Path, str]=None):
		if path is None:
			path = self.CHECKPOINT_SAVEPATH
			self.load_state_dict(th.load(path))
   
	
	def print_model_params(self): 
		print(self.prt_to_tok.weight[0,:])
		print(self.prt.resblocks[0].mlp[0].weight[0,:])
		print(self.vit_to_prt.weight[0,1:20])
		print(self.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
		print(self.vit.conv1.weight[0,:])
		# it would seem that all the model parameters are changing.
  
	def std_model_params(self): 
		q = th.zeros(5)
		q[0] = th.std(self.vit.conv1.weight)
		q[1] = th.std(self.vit.transformer.resblocks[0].mlp[0].weight)
		q[2] = th.std(self.vit_to_prt.weight)
		q[3] = th.std(self.prt.resblocks[0].mlp[0].weight)
		q[4] = th.std(self.prt_to_tok.weight)
		return q