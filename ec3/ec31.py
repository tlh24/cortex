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
from collections import namedtuple

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
image_count = 5*2048 # how many images to keep around
train_iters = 100000
learning_rate = 0.0005 # *starting* learning rate. scheduled.
weight_decay = 5e-6
		
g_logEn = False
toklen = 30
poslen = 6
p_indim = toklen + 1 + poslen*2 
	# extra index: indicates if string has been edited.
e_indim = 5 + toklen + poslen*2
p_ctx = 36
e_ctx = 5
patch_size = 5
v_ctx = int((image_resolution / patch_size) ** 2 + 1)
batch_size = 24
vision_width = 256
prog_width = 128
vision_heads = 8
vision_layers = 4
prog_heads = 8
prog_layers = 6
embed_dim = 256

# positional encoding. 
posenc = th.zeros(p_ctx, poslen*2)
for i in range(p_ctx): 
	for j in range(poslen):
		posenc[i,j*2+0] = sin((2*pi*i / p_ctx) * (j+1))
		posenc[i,j*2+1] = cos((2*pi*i / p_ctx) * (j+1))
		
posenc = posenc.expand(batch_size, p_ctx, poslen*2) 

# batch result datastructure
Bres = namedtuple("Bres", "a_pid b_pid a_progenc b_progenc c_progenc a_progstr b_progstr edits")
	

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

def cost_progenc( progenc ): 
	cost = 0
	for i in range(len(progenc)): 
		cost = cost + ord(progenc[i])
	return cost

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
		print(sp.stderr.peek())
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
			dist = 25.0
			
		lpr = logod_pb2.Logo_last()
		if dist > 24: 
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
			if cost_progenc(db_progenc[mindex]) > cost_progenc(result.progenc) :
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


def apply_onedit(res, edited, typ, cr, pp): 
	# super important to keep this factorized, 
	# otherwise we risk bugs in that training and test don't change state in the same way. 
	getnew = False;
	la = len(res.a_progenc)
	lc = len(res.c_progenc)
	sl = list(res.c_progenc)
	if typ == "sub": 
		if pp > lc-1: 
			pp = lc-1
		if pp < 0: 
			pp = 0 # indx of removed element = la + pp
		sl[pp] = cr
		edited[la+pp] = 0.6
	if typ == "del": 
		if pp >= 0 and pp < lc: 
			del sl[pp] # interesting python has this sugar
		if pp >= lc-1: 
			pp = lc-2
		if pp < 0: 
			pp = 0 # indx of removed element = la + pp
		ed = edited[la+pp+1 : p_ctx].clone()
		edited[la+pp : p_ctx-1] = ed
		edited[p_ctx-1] = 0
		edited[la+pp] = -1.0 # different mark.. ? 
	if typ == "ins": 
		if pp > lc: 
			pp = lc # you can insert at the end.
		if pp < 0: 
			pp = 0 # indx of removed element = la + pp
		sl.insert(pp, cr)
			# note the semantics for these three operations are different! 
		ed = edited[la+pp : p_ctx-1].clone()
		edited[la+pp+1 : p_ctx] = ed
		edited[la+pp] = 1
	if typ == "fin":
		# print(f"apply_edits: getting a new prog at {j}")
		getnew = True
	res = res._replace(c_progenc = "".join(sl))
	return res, edited, getnew

def apply_edits(result, edited): 
	# everything being mutable makes this much easier.. 
	getnew = []
	for j in range(len(result)): 
		res = result[j]
		if len(res.edits) > 0: 
			e = res.edits.pop(0)
			res,ed2,gn = apply_onedit(res, edited[j], e.typ, e.chr, e.pos)
			result[j] = res
			edited[j] = ed2
			if gn: 
				getnew.append(j)
		else: 
			getnew.append(j)
		result[j] = res
	# replace the expired results.
	if len(getnew) > 0: 
		newres,newedit = new_batch_result(len(getnew))
		for j in range(len(getnew)): 
			k = getnew[j]
			result[k] = newres[j]
			edited[k] = newedit[j]
	return result,edited

def apply_yedits(result, edited, yedit, fins): 
	# this from model output, not ocaml.
	(typ, c, pos) = yedit
	for j in range(len(result)): 
		if(fins[j] < 1): 
			res = result[j]
			res,ed2,gn = apply_onedit(res, edited[j], typ[j], c[j], pos[j])
			result[j] = res
			edited[j] = ed2
			if gn: 
				# print(f"model emitted fin channel {j}; a:{res.a_progenc} b:{res.b_progenc} c:{res.c_progenc}" )
				fins[j] = 2
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
		for i in range(l): 
			c = ord(res.a_progenc[i]) - ord('0')
			if c >= 0 and c < toklen-1: 
				batch_p[j, i, c] = 1
		lc = len(res.c_progenc)
		if lc > 16: 
			print("c_progenc too long:",res.c_progenc)
		for i in range(lc): 
			c = ord(res.c_progenc[i]) - ord('0')
			if c >= 0 and c < toklen-1: 
				batch_p[j, i+l, c] = 1
				batch_p[j, i+l, toklen-1] = 1 
					# indicate this string is 'c', to be edited
		# copy over the edit tags 
		batch_p[j,:, toklen] = edited[j]
		
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
	batch_p[:, 0:l, toklen+1:] = posenc[:, 0:l, :]
	batch_p[:, l:l+lc, toklen+1:] = posenc[:, 0:lc, :]
	
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

	buff = bytearray(8*1024) # careful here ... was 4k
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
	
	# change the edit datastructure
	styp = []
	cl = []
	posl = []
	# change to list of namedtuple so it's editable & we can add extra data fields
	res = []
	for i in range(batch_size):
		r = Bres(result.btch[i].a_pid, # for reference
			  result.btch[i].b_pid,		 # for reference
			  result.btch[i].a_progenc, # doesn't change
			  result.btch[i].b_progenc, # hidden to the model
			  result.btch[i].a_progenc, # edited
			  result.btch[i].a_progstr, # human-readable
			  result.btch[i].b_progstr, # human readable
			  result.btch[i].edits)
		res.append(r)
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
	
	def forward(self, u, batch_a, batch_p): 
		# encode the image (we should only need to do this once??)
		q = th.zeros(6)
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
		x = self.prt(vxpx) # bs, v_ctx * p_ctx, prog_width
		q[4] = th.std(x)
		x = th.reshape(x, (batch_size,-1))
		x = self.prt_to_edit(x)
		q[5] = th.std(x)
		# x = self.ln_post(x) # scale the inputs to softmax
		# x = self.gelu(x)
		x = th.cat((self.tok_softmax(x[:,0:4]),
				  self.tok_softmax(x[:,4:4+toklen]), 
				  x[:,4+toklen:]), dim=1)
		return x,q

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
lossfunc = nn.MSELoss(reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def print_model_params(): 
	print(model.prt_to_tok.weight[0,:])
	print(model.prt.resblocks[0].mlp[0].weight[0,:])
	print(model.vit_to_prt.weight[0,1:20])
	print(model.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
	print(model.vit.conv1.weight[0,:])
	# it would seem that all the model parameters are changing.

def std_model_params(): 
	q = th.zeros(5)
	q[0] = th.std(model.vit.conv1.weight)
	q[1] = th.std(model.vit.transformer.resblocks[0].mlp[0].weight)
	q[2] = th.std(model.vit_to_prt.weight)
	q[3] = th.std(model.prt.resblocks[0].mlp[0].weight)
	q[4] = th.std(model.prt_to_tok.weight)
	return q


def decode_edit(y): 
	typ = th.argmax(y[:,0:4], 1)
	typ = typ.cpu() # type should be th.Long
	bs = y.shape[0]
	styp = []
	cl = []
	posl = [] # keep native - faster.
	for j in range(bs): 
		if typ[j] == 0: 
			styp.append('sub')
		if typ[j] == 1: 
			styp.append('del')
		if typ[j] == 2: 
			styp.append('ins')
		if typ[j] == 3: 
			styp.append('fin')
		c = th.argmax(y[j,4:4+toklen]).cpu()
		c = chr(c + ord('0'))
		cl.append(c)
		pos = y[j,5+toklen:]
		z = y[j,5+toklen:]
		z = th.unsqueeze(z,0)
		z = z.expand(p_ctx, -1)
		cos = th.nn.CosineSimilarity(dim=1)
		pos = cos(posenc[0,:,:], z)
		posl.append(th.argmax(pos).item())
	return (styp, cl, posl) 

	
def compare_edit(result, batch_e, y): 
	e = result[0].edits[0]
	# print("ocaml  : ",e.typ," ",e.chr," ",e.pos) # debug; should be the same as below.
	styp, c, pos = decode_edit(batch_e)
	print("batch_e: ", styp[0]," ",c[0]," ",pos[0])
	styp, c, pos = decode_edit(y)
	print("model_y: ", styp[0]," ",c[0]," ",pos[0])

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

losslog = open("losslog.txt", "w")
lr = learning_rate

for u in range(train_iters):
	batch_a, batch_p, batch_e = result_to_batch(result, edited)
	model.zero_grad()
	y,q = model(u, batch_a, batch_p)
	loss = lossfunc(y,batch_e)
	lossflat = th.sum(loss)
	lossflat.backward()
	th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	optimizer.step()
	
	# thrd.join()
	# batch_a, batch_p, batch_e, result, edited = thrd.output
	slowloss = 0.99*slowloss + 0.01 * lossflat.detach()
	if u % 17 == 0 :
		print(f'{u} {lr} loss: {lossflat}; slowloss {slowloss}')
		print(result[0].a_progenc,"-->", result[0].c_progenc,"(", result[0].b_progenc,")\t", "[",result[0].a_pid,"]",result[0].a_progstr,"-->", "[",result[0].b_pid,"]", result[0].b_progstr)
		compare_edit(result, batch_e, y)
		# print(i, lossflat, loss[0], y_mask[0])
		# print(x[0])
		# print(y[0])
		# print(x[0] - y[0])
	result, edited = apply_edits(result, edited)
	losslog.write(f"{u}\t{slowloss}")
	for i in range(q.shape[0]): 
		losslog.write(f"\t{q[i].cpu().item()}")
	losslog.write("\n")
	losslog.flush()
	
	# change the learning rate. 
	lr = learning_rate
	if u > 1000:
		lr = lr * (1 + ((u-1000) / 5000))
	lr = min(lr, 0.003)
	for g in optimizer.param_groups:
		g['lr'] = lr

	
losslog.close()

# try creating programs, just to see what sort of errors are made. 
result, edited = new_batch_result(batch_size)
for u in range(16): 
	batch_a, batch_p, batch_e = result_to_batch(result, edited)
	y,q = model(u, batch_a, batch_p)
	typ,c,pos = decode_edit(y)
	fins = th.zeros(batch_size)
	result, edited = apply_yedits(result, edited, (typ,c,pos), fins)

# print the results, if it falls through. 
print("resulting decoded programs:")
for j in range(batch_size): 
	res = result[j]
	mtch = 'x'
	if res.b_progenc == res.c_progenc: 
		mtch = ''
	print(f"channel {j}; a:{res.a_progenc} b:{res.b_progenc} c:{res.c_progenc} {mtch}" )

# # see if it can create working programs
# softmx = nn.Softmax(dim = 1)
# batch_a, batch_p, batch_pl = make_batch()
# batch_p_orig = batch_p.clone()
# for i in range(int(np.max(batch_pl))):
# 	mask, y, y_mask = make_mask_y(i, batch_p, batch_pl)
# 	x = model(batch_a, batch_p, mask)
# 	# substitute this into batch_p
# 	x = softmx(x[:, 0:toklen])
# 	indx = th.argmax(x, dim=1)
# 	batch_p[:, i] = th.zeros(p_indim)
# 	batch_p[:, i, indx] = 1.0
# 	for j in range(batch_size): 
# 		typ = tktype[indx[j]]
# 		batch_p[j, i, toklen+typ] = 1.0
# 
# # now process it back to human-readable.
# for j in range(batch_size): 
# 	prog_orig = []
# 	prog_new = []
# 	for i in range(int(np.max(batch_pl))):
# 		prog_orig.append(th.argmax(batch_p_orig[j, i, 0:toklen]))
# 		prog_new.append(th.argmax(batch_p[j, i, 0:toklen]))
# 	prog_orig_h = prog_to_human(prog_orig)
# 	prog_new_h = prog_to_human(prog_new)
# 	print(f"batch{j}\n\torig {prog_orig_h}\n\tnew {prog_new_h}\n")

logf.close()
sp.terminate()
sys.exit(0)
