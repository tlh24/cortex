import math
from math import sin, cos, pi
import torch as th
from torch import nn, optim
import fcntl, os, select, sys
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import subprocess 
import time
from threading import Thread

import logod_pb2
import xf
import clip_model

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')

subprocess.run(["mkdir","-p","/tmp/png"]) # required by ocaml

ocamlLogoPath = "./_build/default/program.exe"
sp = subprocess.Popen(ocamlLogoPath, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=False)

def make_nonblock(fd): 
	fl = fcntl.fcntl(fd, fcntl.F_GETFL)
	fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
	
make_nonblock(sp.stdout.fileno())
make_nonblock(sp.stderr.fileno())


image_resolution = 30
image_count = 2*2048 # how many images to keep around
		
g_logEn = False
toklen = 30
poslen = 6
p_indim = toklen + 1 + poslen*2 
	# the 1 indicates if it's been edited.
e_indim = 5 + toklen + poslen*2
p_ctx = 50
e_ctx = 5
patch_size = 5
v_ctx = int((image_resolution / patch_size) ** 2 + 1)
batch_size = 24
prog_width = 128
# xfrmr = xf.Transformer(p_ctx, prog_width, 8, 8)
# 	# p_ctx, width, layers, heads
# encoder = nn.Linear(p_indim, prog_width)
# 	# just a simple linear layer to put into transformer latent space. 
# 	# how are these weights initialized?? 
# gelu = nn.GELU()

# positional encoding. 
posenc = th.zeros(p_ctx, poslen*2)
for i in range(p_ctx): 
	for j in range(poslen):
		posenc[i,j*2+0] = sin((2*pi*i / p_ctx) * (j+1))
		posenc[i,j*2+1] = cos((2*pi*i / p_ctx) * (j+1))
		
posenc = posenc.expand(batch_size, p_ctx, poslen*2) 

# prog = [0 for i in range(p_ctx)]
# prog_human = list(map(lambda i : tokens[i], prog))
# x = th.zeros(p_ctx, p_indim)
# for i in range(p_ctx): # might be faster way? 
# 	x[i, prog[i]] = 1.0
# 	typ = tktype[prog[i]]
# 	x[i, toklen+typ] = 1.0
# 	# add positional encoding too. 
# 
# x = x.expand(batch_size, p_ctx, p_indim).contiguous()
# x[:,:,toklen : p_indim] = posenc
# x = encoder(x)
# x = gelu(x)
# x = xfrmr(x)
# that works pretty seamlessly! 
# x.shape is now [1, 32, 128] 
# -- the output of a decoder section
# now need to add a decoder section to output tokens + continuations. 
# and then do search to boostrap..
# no, see cortex_bumps.py; we can start with an editor. 

# next task is to enumerate programs ('templates') then mess 'em up and ask the transformer to fix (or not)
# I guess do n-level depth-first search
logf = open("ec3_log.txt", "w")

def prog_to_string(prog): 
	q = list(map(lambda i : tokens[i], prog))
	q = " ".join(q)
	q = "( " + q + " )"
	return q
	
def check_formed(prog) : 
	q = prog_to_string(prog)
	return ocaml_run(q)
	
def prog_to_human(prog): 
	return list(map(lambda i : tokens[i], prog))


db_img = th.ones(image_count, image_resolution, image_resolution)*-100 # prevent spurious matches
db_prog = []
db_progenc = []
db_segs = []
db_cnt = 0
ocount = 0

num_rejections = 0
num_replacements = 0

while db_cnt < image_count : 
	ocount = ocount + 1
	lp = logod_pb2.Logo_request()
	lp.id = db_cnt
	lp.log_en = g_logEn
	lp.res = image_resolution
	lp.batch = 0
	q = lp.SerializeToString()
	sp.stdin.write(q)
	sp.stdin.flush()
	#print("requesting a new program + image pair; bytes",len(q))
	
	streams = [ sp.stdout ]
	temp0 = []
	readable, writable, exceptional = select.select(streams, temp0, temp0, 5)
	if len(readable) == 0:
		raise Exception("Timeout of 5 seconds reached!")
	
	buff = bytearray(1024)
	nrx = sp.stdout.readinto(buff)
	if nrx <= 0:
		print("No data received on stdout! stderr: ", sp.stderr.peek())
	# convert the bytearray to a protobuf.. 
	result = logod_pb2.Logo_result()
	try: 
		result.ParseFromString(bytes(buff[0:nrx]))
		# print(result)
	except Exception as inst: 
		print("ParseFromString; ", nrx, buff[0:nrx], "stderr:", sp.stderr.peek())
		
	buff2 = bytearray(1500)
	n = sp.stderr.readinto(buff2)
	if n != result.stride * result.height: 
		print(ocount,"unexpected number of bytes in image, ", n)
		print(ocount,"stride ", result.stride, " height ", result.height)
	else: 
		# do something else with this image..
		a = np.frombuffer(buff2, dtype=np.uint8, count=n)
		a = np.reshape(a, (result.height, result.stride))
		a = a[:, 0:result.width]
		a = th.tensor(a)
		if db_cnt > 0 : 
			d = th.sum((db_img - a)**2, (1,2))
			mindex = th.argmin(d)
			dist = d[mindex]
		else: 
			mindex = 0
			dist = 15.0
			
		lpr = logod_pb2.Logo_last()
		if dist > 10: 
			# add to the database. 
			db_img[db_cnt, :, :] = a
			db_prog.append(result.prog)
			db_progenc.append(result.progenc)
			db_segs.append(result.segs) # used to be "reverse()"
			
			lpr.keep = True
			lpr.where = db_cnt
			lpr.render_simplest = (db_cnt == image_count-1)
			
			print(db_cnt, num_rejections, num_replacements, result.prog)
			
			db_cnt = db_cnt+1
			
			if False: 
				plt.imshow(a.cpu().numpy())
				plt.colorbar()
				plt.show()
		else: 
			num_rejections = num_rejections+1
			# reject the more complex representation.
			if len(db_progenc[mindex]) > len(result.progenc) :
				print(f"replacing {mindex} with {db_cnt}")
				lpr.keep = True
				lpr.where = mindex
				db_img[mindex, :, :] = a
				db_prog[mindex] = result.prog
				db_progenc[mindex] = result.progenc
				db_segs[mindex] = result.segs # used to be reverse
				num_replacements += 1
			else: 
				lpr.keep = False
				lpr.where = -1
		q = lpr.SerializeToString()
		#print("responding; bytes", len(q))
		sp.stdin.write(q)
		sp.stdin.flush()
		
		streams = [ sp.stdout ]
		temp0 = []
		readable, writable, exceptional = select.select(streams, temp0, temp0, 5)
		if len(readable) == 0:
			raise Exception("Timeout of 5 seconds reached!")
		
		nrx = sp.stdout.readinto(buff)
		if nrx <= 0:
			print("No data received on stdout! stderr: ", sp.stderr.peek())
		result = logod_pb2.Logo_ack()
		try: 
			result.ParseFromString(bytes(buff[0:nrx]))
			# print(result)
		except Exception as inst: 
			print("ParseFromString; ", nrx, buff[0:nrx], "stderr:", sp.stderr.peek())


print(f"done with {image_count} unique image-program pairs")
print(f"there were {num_rejections} rejections due to image space collisions")
print(f"of these, {num_replacements} were simplifications")

print("first 10 programs:")
for j in range(10): 
	print(j, ": ", db_prog[j])

# what we need to do is train with a series of edits, 
# # and replace the entries that have already emitted 'done'. 

def apply_edits(result,edited): 
	# everything being mutable makes this much easier.. 
	getnew = []
	for j in range(len(result)): 
		res = result[j]
		pl = len(res.a_progenc)
		if len(res.edits) > 0: 
			e = res.edits.pop(0)
			sl = list(res.a_progenc)
			if e.typ == "sub": 
				sl[e.pos] = e.chr
				edited[j][e.pos] = 1
				edited[j][e.pos+pl+1] = 1
			if e.typ == "del": 
				del sl[e.pos] # interesting python has this sugar
				edited[j][e.pos] = 1
			if e.typ == "ins": 
				sl.insert(e.pos, e.chr)
				edited[j][e.pos+pl+1] = 1
				# also note the semantics for these three operations are different! 
			if e.typ == "fin":
				# print(f"apply_edits: getting a new prog at {j}")
				getnew.append(j)
			res.a_progenc = "".join(sl)
		else: 
			getnew.append(j)
	# replace the expired results.
	if len(getnew) > 0: 
		newres,newedit = new_batch_result(len(getnew))
		for j in range(len(getnew)): 
			k = getnew[j]
			result[k] = newres[j]
			edited[k] = newedit[j]
	return result,edited

def result_to_batch(result, edited): 
	# convert the results datastructure (as partly returned from ocaml / protobufs) into torch tensors. 
	batch_a = th.zeros(batch_size, 3, image_resolution, image_resolution)
	batch_p = th.zeros(batch_size, p_ctx, p_indim)
	batch_e = th.zeros(batch_size, e_indim)
	for j in range(len(result)): 
		res = result[j]
		batch_a[j,0,:,:] = db_img[res.a_pid, :, :]
		batch_a[j,1,:,:] = db_img[res.b_pid, :, :]
		batch_a[j,2,:,:] = batch_a[j,0,:,:] - batch_a[j,1,:,:]
		# print("result_to_batch: python database for programs:")
		# print(res.a_pid, db_prog[res.a_pid])
		# print(res.b_pid, db_prog[res.b_pid])
		# encode the input string twice -- the second time is for editing (& will need to be redone during the training )
		l = len(res.a_progenc)
		if l > 16: 
			print("too long:",res.a_progenc,res.a_progstr)
		for i in range(len(res.a_progenc)): 
			c = ord(res.a_progenc[i]) - ord('0')
			if c >= 0 and c < toklen: 
				batch_p[j, i, c] = 1
				batch_p[j, i+l+1, c] = 1
			# delimeter
			batch_p[j, l, toklen-1] = 1
		# copy over the edit tags 
		batch_p[j,:,toklen] = edited[j]
		
		e = res.edits[0]
		if e.typ == "sub": 
			batch_e[j,0] = 1
		if e.typ == "del": 
			batch_e[j,1] = 1
		if e.typ == "ins": 
			batch_e[j,2] = 1
		if e.typ == "fin": 
			batch_e[j,3] = 1
		c = ord(e.chr) - ord('0')
		if c >= 0 and c < toklen: 
			batch_e[j,c+4] = 1
		#position encoding
		ofst = 5 + toklen
		batch_e[j,ofst:] = posenc[0,e.pos,:]

	# add positional encoding to program one-hot
	batch_p[:, :, toklen+1:] = posenc
	
	return batch_a, batch_p, batch_e

def new_batch_result(batch_size) : 
	# new batch data from ocaml. 
	lp = logod_pb2.Logo_request()
	lp.id = 0
	lp.log_en = g_logEn
	lp.res = 0
	lp.batch = batch_size
	q = lp.SerializeToString()
	sp.stdin.write(q)
	sp.stdin.flush()

	streams = [ sp.stdout ]
	temp0 = []
	readable, writable, exceptional = select.select(streams, temp0, temp0, 5)
	if len(readable) == 0:
		raise Exception("Timeout of 5 seconds reached!")

	buff = bytearray(4*1024)
	nrx = sp.stdout.readinto(buff)
	if nrx <= 0:
		print("No data received on stdout! stderr: ", sp.stderr.peek())
	# convert the bytearray to a protobuf.. 
	result = logod_pb2.Logo_batch()
	try: 
		result.ParseFromString(bytes(buff[0:nrx]))
		# print(result)
	except Exception as inst: 
		print("ParseFromString; ", nrx, buff[0:nrx], "stderr:", sp.stderr.peek())
	# print("batch result count:", result.count)
	# for res in result.btch: 
	# 	print(res.a_pid, res.a_progenc, res.a_progstr) 
	# 	print(res.b_pid, res.b_progenc, res.b_progstr)
	# 	for e in res.edits: 
	# 		print("edit ", e.typ, e.pos, e.chr)
	# sys.stdout.flush()
	
	# change to list so it's editable
	res = []
	for i in range(batch_size):
		res.append(result.btch[i])
	result = res
	# indicate that these are unedited. 
	edited = []
	for j in range(batch_size): 
		edited.append(th.zeros(p_ctx))
	
	return result,edited

# result, edited = new_batch_result(batch_size)
# for j in range(10): 
# 	print("===", j)
# 	print(result)
# 	batch_a, batch_p, batch_e = result_to_batch(result, edited)
# 	fig, axs = plt.subplots(1,3, figsize=(13,5))
# 	axs[0].imshow(batch_a[0,0,:,:].cpu().numpy())
# 	axs[1].imshow(batch_a[0,1,:,:].cpu().numpy())
# 	axs[2].imshow(batch_p[0,:,:].cpu().numpy())
# 	plt.show()
# 	result, edited = apply_edits(result, edited)
# 	print("after edit:")
# 	print(result)
# 	
# print("=== done for now ===")

def build_attention_mask2(v_ctx, p_ctx):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    ctx = v_ctx + p_ctx
    mask = th.empty(ctx, ctx)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    mask[0:v_ctx, 0:v_ctx] = 1.0
    return mask

# alrighty then!  time to train the transformer to create these strings! 
# decoder-only architecture, input is the image (compressed) and previous tokens. 
vision_width = 256
vision_heads = 8
vision_layers = 4
prog_heads = 8
prog_layers = 6
embed_dim = 256

class ecTransformer(nn.Module):
	def __init__(self, image_resolution: int, vision_width:int, patch_size:int,  prog_width:int, embed_dim:int, v_ctx:int, p_ctx:int, p_indim:int, e_indim:int): 
		super().__init__()
		self.v_ctx = v_ctx
		self.p_ctx = p_ctx
		self.p_indim = p_indim
		
		self.vit = clip_model.VisionTransformer(
			input_resolution = image_resolution, 
			patch_size = patch_size, 
			width = vision_width, 
			layers = 4, 
			heads = 8, 
			output_dim = embed_dim)

		self.vit_to_prt = nn.Linear(embed_dim, prog_width)
		
		
		self.encoder = nn.Linear(p_indim, prog_width)
		self.prt = clip_model.Transformer(
			width = prog_width, 
			layers = 6, 
			heads = 8, 
			attn_mask = build_attention_mask2(v_ctx, p_ctx))
			
		self.prt_to_edit = nn.Linear(prog_width * (p_ctx + v_ctx), e_indim)
		self.ln_post = clip_model.LayerNorm(e_indim)
		self.gelu = clip_model.QuickGELU()
		self.tok_softmax = nn.Softmax(dim = 1)
	
	def forward(self, batch_a, batch_p): 
		# encode the image (we should only need to do this once??)
		vx = self.vit(batch_a) # x is size [16, v_ctx, 256] 
		vx = self.vit_to_prt(vx)
		# vx = gelu(vx) # ? needed ? 

		px = self.encoder(batch_p)
		vxpx = th.cat((vx, px), dim = 1)
		# x = vxpx * mask
		x = self.prt(vxpx) # bs, v_ctx * p_ctx, prog_width
		x = th.reshape(x, (batch_size,-1))
		x = self.prt_to_edit(x)
		x = self.ln_post(x) # scale the inputs to softmax
		x = self.gelu(x)
		# x = self.tok_softmax(x)
		return x

model = ecTransformer(image_resolution = image_resolution, 
							 vision_width = vision_width, 
							 patch_size = patch_size, 
							 prog_width = prog_width, 
							 embed_dim = embed_dim, 
							 v_ctx = v_ctx, 
							 p_ctx = p_ctx, 
							 p_indim = p_indim, 
							 e_indim = e_indim)

# lossfunc = nn.CrossEntropyLoss(label_smoothing = 0.08, reduction='none')
lossfunc = nn.MSELoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

def print_model_params(): 
	print(model.prt_to_tok.weight[0,:])
	print(model.prt.resblocks[0].mlp[0].weight[0,:])
	print(model.vit_to_prt.weight[0,1:20])
	print(model.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
	print(model.vit.conv1.weight[0,:])
	# it would seem that all the model parameters are changing.

# def make_batch(): 
# 	indx = np.random.choice(len(valid_nonzero), size=batch_size, replace=False)
# 	batch_a = th.zeros((batch_size, image_resolution, image_resolution)) # todo: fp16
# 	batch_p = th.zeros((batch_size, p_ctx, p_indim))
# 	batch_pl = np.zeros(batch_size)
# 	j = 0
# 	for i in indx: 
# 		prog, a = valid_nonzero[i] # note! this is a reference. 
# 		batch_a[j,:,:] = th.tensor(a)
# 		# encode the program, too
# 		prog = prog + [toklen-1] # terminate; return new list.
# 		batch_pl[j] = len(prog)
# 		# prog_human = list(map(lambda i : tokens[i], prog))
# 		for k in range(len(prog)): 
# 			batch_p[j, k, prog[k]] = 1.0
# 			typ = tktype[prog[k]]
# 			batch_p[j, k, toklen+typ] = 1.0
# 		j = j+1
# 		
# 	batch_p[:,:,toklen+typlen : p_indim] = posenc
# 	batch_a = th.unsqueeze(batch_a, dim = 1) # batch, channels, H,
# 	return (batch_a, batch_p, batch_pl)
# 
# def make_mask_y(i, batch_p, batch_pl): 
# 	y = batch_p[:, i, 0:toklen + typlen]
# 	y_mask = th.tensor(batch_pl > i).detach() 
# 	mask = th.zeros(batch_size, v_ctx + p_ctx, prog_width)
# 	mask[:, 0:v_ctx, :] = 1.0
# 	if i > 1: 
# 		mask[:, v_ctx:v_ctx+i, :] = 1.0
# 	return (mask, y, y_mask)

def decode_edit(y): 
	typ = th.argmax(y[0,0:4])
	typ = typ.cpu()
	if typ == 0: 
		styp = 'sub'
	if typ == 1: 
		styp = 'del'
	if typ == 2: 
		styp = 'ins'
	if typ == 3: 
		styp = 'fin'
	c = th.argmax(y[0,4:4+toklen]).cpu()
	c = chr(c + ord('0'))
	pos = y[0,5+toklen:]
	z = y[0,5+toklen:]
	z = th.unsqueeze(z,0)
	z = z.expand(p_ctx, -1)
	cos = th.nn.CosineSimilarity(dim=1)
	pos = cos(posenc[0,:,:], z)
	pos = th.argmax(pos).item()
	return (styp, c, pos) 
	
def compare_edit(result, batch_e, y): 
	e = result[0].edits[0]
	print("ocaml  : ",e.typ," ",e.chr," ",e.pos)
	styp, c, pos = decode_edit(batch_e)
	print("batch_e: ", styp," ",c," ",pos)
	styp, c, pos = decode_edit(y)
	print("model_y: ", styp," ",c," ",pos)

class SimpleThread(Thread):
	# this just became confusing due to python's data model..
	def __init__(self, result, edited):
		super().__init__()
		self.result = result
		self.edited = edited
		self.output = None

	def run(self):
		batch_a, batch_p, batch_e = result_to_batch(self.result, self.edited)
		result, edited = apply_edits(self.result, self.edited)
		# apply_edits also gets new data
		self.output = batch_a, batch_p, batch_e, result, edited

slowloss = 0.0
result, edited = new_batch_result(batch_size)
batch_a, batch_p, batch_e = result_to_batch(result, edited)

compare_edit(result, batch_e, batch_e)
result, edited = apply_edits(result, edited)
batch_a, batch_p, batch_e = result_to_batch(result, edited)
compare_edit(result, batch_e, batch_e)
result, edited = apply_edits(result, edited)
batch_a, batch_p, batch_e = result_to_batch(result, edited)
compare_edit(result, batch_e, batch_e)
print("=====")

for u in range(100000): 
	batch_a, batch_p, batch_e = result_to_batch(result, edited)
	# thrd = SimpleThread(result, edited)
	# thrd.start()
	model.zero_grad()
	y = model(batch_a, batch_p)
	loss = lossfunc(y,batch_e)
	lossflat = th.sum(loss)
	lossflat.backward()
	optimizer.step()
	
	# thrd.join()
	# batch_a, batch_p, batch_e, result, edited = thrd.output
	slowloss = 0.99*slowloss + 0.01 * lossflat.detach()
	if u % 20 == 0 :
		print(f'{u} loss: {lossflat}; slowloss {slowloss}')
		print(result[0].a_progstr,"-->", result[0].b_progstr)
		compare_edit(result, batch_e, y)
		# print(i, lossflat, loss[0], y_mask[0])
		# print(x[0])
		# print(y[0])
		# print(x[0] - y[0])
	result, edited = apply_edits(result, edited)

# see if it can create working programs
softmx = nn.Softmax(dim = 1)
batch_a, batch_p, batch_pl = make_batch()
batch_p_orig = batch_p.clone()
for i in range(int(np.max(batch_pl))):
	mask, y, y_mask = make_mask_y(i, batch_p, batch_pl)
	x = model(batch_a, batch_p, mask)
	# substitute this into batch_p
	x = softmx(x[:, 0:toklen])
	indx = th.argmax(x, dim=1)
	batch_p[:, i] = th.zeros(p_indim)
	batch_p[:, i, indx] = 1.0
	for j in range(batch_size): 
		typ = tktype[indx[j]]
		batch_p[j, i, toklen+typ] = 1.0

# now process it back to human-readable.
for j in range(batch_size): 
	prog_orig = []
	prog_new = []
	for i in range(int(np.max(batch_pl))):
		prog_orig.append(th.argmax(batch_p_orig[j, i, 0:toklen]))
		prog_new.append(th.argmax(batch_p[j, i, 0:toklen]))
	prog_orig_h = prog_to_human(prog_orig)
	prog_new_h = prog_to_human(prog_new)
	print(f"batch{j}\n\torig {prog_orig_h}\n\tnew {prog_new_h}\n")

logf.close()
sp.terminate()
sys.exit(0)
