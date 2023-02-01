import math
import mmap
import numpy as np
import torch as th
import argparse
import matplotlib.pyplot as plt
from ctypes import * # for c_char
import time

def make_mmf_rd(fname): 
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)
	
def make_mmf_wr(fname): 
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)

def read_mmap(mmf, dims): 
	mmf.seek(0)
	mmb = mmf.read()
	siz = len(mmb)
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
	
fd_bpro = make_mmf_rd("bpro_0.mmap")
fd_bimg = make_mmf_rd("bimg_0.mmap")
fd_bedts = make_mmf_rd("bedts_0.mmap")
fd_bedtd = make_mmf_wr("bedtd_0.mmap")
fd_posenc = make_mmf_rd("posenc_0.mmap")

parser = argparse.ArgumentParser(description='image mmaped files')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
args = parser.parse_args()
batch_size = args.batch_size
print(f"batch_size:{batch_size}")

image_res = 30
toklen = 30
poslen = 6
p_indim = toklen + 1 + poslen*2 
e_indim = 5 + toklen + poslen*2
p_ctx = 64


plot_rows = 2
plot_cols = 3
figsize = (18, 10)
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

while True:
	bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
	bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
	bedts = read_mmap(fd_bedts, [batch_size, e_indim])
	bedtd = read_mmap(fd_bedtd, [batch_size, e_indim])
	posenc = read_mmap(fd_posenc, [p_ctx, poslen*2])

	plot_tensor(0, 0, bpro[0,:,:], "bpro[0,:,:]", -1.0, 1.0)
	plot_tensor(0, 1, bimg[0,0,:,:], "bimg[0,:,:]", -1.0, 1.0)
	plot_tensor(0, 2, bimg[0,1,:,:], "bimg[1,:,:]", -1.0, 1.0)
	plot_tensor(1, 0, bedts[:,:], "bedts[:,:]", -1.0, 1.0)
	plot_tensor(1, 1, bedtd[:,:], "bedtd[:,:]", -1.0, 1.0)
	plot_tensor(1, 2, posenc[:,:], "posenc[:,:]", -1.0, 1.0)
	
	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(1)
	print("tick")
	initialized=True

