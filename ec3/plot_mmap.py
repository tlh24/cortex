import math
import mmap
import numpy as np
import torch as th
import argparse
import matplotlib.pyplot as plt
from ctypes import * # for c_char
import time
import os

from constants import *
# remove menubar buttons
plt.rcParams['toolbar'] = 'None'

def make_mmf(fname): 
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)

def read_mmap(mmf, dims): 
	mmf.seek(0)
	mmb = mmf.read()
	# siz = len(mmb)
	siz = math.prod(dims) * 4
	mmb2 = (c_char * siz).from_buffer_copy(mmb)
	x = th.frombuffer(mmb2, dtype=th.float).clone()
	x = th.reshape(x, dims)
	return x
	
def write_mmap(mmf, data): 
	buff = io.BytesIO()
	torch.save(data, buff)
	buff.seek(0)
	mmf.seek(0)
	n = mmf.write(buff.read())
	return n


parser = argparse.ArgumentParser(description='image mmaped files')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
g_dreaming = args.dreaming
if args.dreaming: 
	filno = 1
else: 
	filno = 0
print(f"batch_size:{batch_size}")

if not os.path.exists(f"editdiff_{filno}.mmap"): 
	os.system(f'fallocate -l {batch_size*e_indim*4} editdiff_{filno}.mmap')


fd_bpro = make_mmf(f"bpro_{filno}.mmap")
fd_bimg = make_mmf(f"bimg_{filno}.mmap")
fd_bedts = make_mmf(f"bedts_{filno}.mmap")
fd_bedtd = make_mmf(f"bedtd_{filno}.mmap")
fd_posenc = make_mmf(f"posenc_{filno}.mmap")
fd_editdiff = make_mmf(f"editdiff_{filno}.mmap")


# fallocate -l 6016 editdiff_0.mmap for batch size 32
# 6016 = 32 * 47 * 4


plot_rows = 2
plot_cols = 3
figsize = (22, 11)
plt.ion()
fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]


def plot_tensor(r, c, v, name, lo, hi):
	if not initialized:
		# seed with random data so we get the range right
		cmap_name = 'PuRd' # purple-red
		if lo == -1*hi:
			cmap_name = 'seismic'
		data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
		data = np.reshape(data, (v.shape[0], v.shape[1]))
		im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
		cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
	im[r][c].set_data(v.numpy())
	#cbar[r][c].update_normal(im[r][c]) # probably does nothing
	axs[r,c].set_title(name)

bs = batch_size
if batch_size > 32: 
	bs = 32

while True:
	bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
	bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
	bedts = read_mmap(fd_bedts, [batch_size, e_indim])
	bedtd = read_mmap(fd_bedtd, [batch_size, e_indim])
	posenc = read_mmap(fd_posenc, [p_ctx, poslen])
	editdiff = read_mmap(fd_editdiff, [batch_size, e_indim])

	plot_tensor(0, 0, bpro[0,:,:], "bpro[0,:,:]", -2.0, 2.0)
	plot_tensor(0, 1, bimg[0,0,:,:], "bimg[0,0,:,:]", -1.0, 1.0)
	plot_tensor(0, 2, bimg[0,1,:,:], "bimg[0,1,:,:]", -1.0, 1.0)
	plot_tensor(1, 0, bedts[:bs,:], "bedts[:,:]", -2.0, 2.0)
	plot_tensor(1, 1, bedtd[:bs,:], "bedtd[:,:]", -2.0, 2.0)
	plot_tensor(1, 2, editdiff[:bs,:], "editdiff", -2.0, 2.0) # brighter colors
	
	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	# time.sleep(2)
	print("tick")
	initialized=True

