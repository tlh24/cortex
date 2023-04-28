import torch as th
from torch import nn, optim
import torch.cuda.amp
import time
import argparse
import os
from data_exchange import SocketClient, MemmapOrchestrator
from config import ModelConfig
from recognizer.model import Recognizer

import torch._dynamo as dynamo
dynamo.config.verbose=True
# note: I can't seem to get this to work. tlh April 7 2023


parser = argparse.ArgumentParser(description='Transformer-based program synthesizer')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
dreaming = args.dreaming


mc = ModelConfig(dreaming=args.dreaming, batch_size=args.batch_size)

print(f"batch_size:{mc.batch_size}")
print(f"dreaming:{mc.dreaming}")

# setup the socket client and handshake with the server.
socket_client = SocketClient(mc.dreaming)
socket_client.connect()
# socket_client.handshake() -- not needed anymore!  April 27 2023


os.system(mc.edsiz_allocate_command)
# the other mmaps are allocated by ocaml.
	
mo = MemmapOrchestrator(
	mmapno=mc.mmapno, 
	p_ctx=mc.p_ctx, 
	poslen=mc.poslen,
	batch_size=mc.batch_size,
	p_indim=mc.p_indim,
	image_res=mc.image_res,
	e_indim=mc.e_indim,
)


torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_float32_matmul_precision('high') # desktop.



model = Recognizer(
	image_resolution = mc.image_res, 
	vision_width = mc.vision_width, 
	patch_size = mc.patch_size, 
	prog_width = mc.prog_width, 
	embed_dim = mc.embed_dim, 
	v_ctx = mc.v_ctx, 
	p_ctx = mc.p_ctx, 
	p_indim = mc.p_indim, 
	e_indim = mc.e_indim
 )

model.print_n_params()

from os.path import exists
fname = "checkpoints/recognizer_checkpoint.ptx"
if exists(fname): 
	loaded_dict = torch.load(fname)
	model.load_state_dict(loaded_dict)
	print(f"loaded {fname}")
else: 
	if exists("ec32.ptx"):
		loaded_dict = torch.load("ec32.ptx")
		model.load_state_dict(loaded_dict)

# model = nn.DataParallel(model)

# loss is on the predicted edit: 
# [0:4] is the categorical edit type, sub del ins fin
# [4:toklen] is the categorical character/token.  
# [5+toklen:5+toklen+poslen] is the (absolute) position encoding, vectoral.
# not sure why there is a 5 in there.

lossfunc_cel = nn.CrossEntropyLoss(label_smoothing = 0.08, reduction='mean')
lossfunc_mse = nn.MSELoss(reduction='mean')
# !SGD does not work!  AdamW much better.  
optimizer = optim.Adam(model.parameters(), lr=mc.learning_rate)

	
scaler = torch.cuda.amp.GradScaler()
slowloss = 1.0
losslog = open("loss_log.txt", "w")
tic = time.time()
if mc.training:
	print("training...")
if mc.dreaming:
	print("dreaming...")
	torch. set_grad_enabled(False)

# compiling this does not seem to work... 
def train(mod, bimg, bpro, bedts): 
	model.zero_grad()
	y,q = model(u, bimg, bpro)
	loss = lossfunc_mse(y, bedts)
	lossflat = th.sum(loss)
	lossflat.backward()
	th.nn.utils.clip_grad_norm_(model.parameters(), 0.025)
	optimizer.step()
	return y,q,lossflat
	

for u in range(mc.train_iters): 
	# keep things synchronous for now. 
	socket_client.send_and_receive(message="update_batch")
	
	bpro = mo.read_bpro()
	bimg = mo.read_bimg()
	bedts = mo.read_bedts()

	if th.min(bedts[:,0]) < 0: 
		print("bedts synchronization issue!")
	
	if mc.training: 
		model.zero_grad()
		y,q = model(u, bimg.cuda(), bpro.cuda())
		targ = bedts.cuda()
		loss = lossfunc_mse(y, targ)
		lossflat = th.sum(loss)
		lossflat.backward()
		th.nn.utils.clip_grad_norm_(model.parameters(), 0.025)
		optimizer.step() 
		lossflat.detach()
	else: 
		bimg = bimg.cuda()
		bpro = bpro.cuda()
		# introduce noise to the target image and average model output 
		# so ocaml has a better estimate of edit probabilities
		bimg1 = bimg + th.randn(batch_size, 3, mc.image_res, mc.image_res) * 0.1
		y1,q = model(u, bimg1, bpro)
		bimg2 = bimg + th.randn(batch_size, 3, mc.image_res, mc.image_res) * 0.1
		y2,q = model(u, bimg2, bpro)
		bimg3 = bimg + th.randn(batch_size, 3, mc.image_res, mc.image_res) * 0.1
		y3,q = model(u, bimg3, bpro)
		y = (y1 + y2 + y3)/3.0
		lossflat = 0.0
  
	slowloss = 0.99*slowloss + 0.01 * lossflat
 
	if mc.training: 
		losslog.write(f"{u}\t{slowloss}")
		for i in range(q.shape[0]): 
			losslog.write(f"\t{q[i].cpu().item()}")
		losslog.write(f"\t{mc.nreplace+0.001}")
		losslog.write("\n")
		losslog.flush()
	
	mo.write_bedtd(y)
	if mc.training: 
		mo.write_editdiff(bedts - y.cpu()) # synchronization.
	socket_client.send_and_receive(message="decode_edit")
  
	if u % 11 == 0 :
		toc = time.time()
		rate = int((batch_size * 11) / (toc - tic))
		tic = toc
		print(f'{u} {mc.learning_rate:.6f} loss: {lossflat:.5f}; slowloss {slowloss:.5f}; {rate} samp/sec')
				
	if u % 1000 == 999 : 
		if mc.training: 
			model.save_checkpoint()
		if mc.dreaming: 
			model.load_checkpoint()
			print("dreamer reloaded model parameters.")
	


mo.close()
socket_client.close()

