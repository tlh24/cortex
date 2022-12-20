import mmap
import torch as th
import matplotlib.pyplot as plt
import copy
from ctypes import *
import socket
import time

batch_size = 24
image_res = 30
batch_size = 24
toklen = 30
poslen = 6
p_indim = toklen + 1 + poslen*2 
e_indim = 5 + toklen + poslen*2
p_ctx = 36

def make_mmf(fname): 
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
	
fd_bpro = make_mmf("bpro.mmap")
fd_bimg = make_mmf("bimg.mmap")
fd_bedt = make_mmf("bedt.mmap")
fd_posenc = make_mmf("posenc.mmap")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 4340))

plot_rows = 2
plot_cols = 2
figsize = (12,8)
im = [ [0]*plot_cols for i in range(plot_rows)]
plt.ion()
fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)

for u in range(150): 
	
	sock.sendall(b"update_batch\n")
	data = sock.recv(1024)
	print(f"Received {data!r}")
	
	bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
	bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
	bedt = read_mmap(fd_bedt, [batch_size, e_indim])
	posenc = read_mmap(fd_posenc, [p_ctx, poslen*2])
	
	def disp_mtrx(r, c, m, titl): 
		if u == 0 : 
			im[r][c] = axs[r,c].imshow(m.numpy())
		else: 
			im[r][c].set_data(m.numpy())
		axs[r,c].set_title(titl)
	
	for i in range(1):
		disp_mtrx(0,0,bpro[i,:,:], f'bpro {i}')

	for i in range(1):
		disp_mtrx(0,1,bimg[i,2,:,:], f'bimg {i}')

	disp_mtrx(1,0,bedt[:,:], f'bedt {i}')

	disp_mtrx(1,1,posenc[:,:], f'posenc {i}')
	
	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(1)
	

fd_bpro.close()
fd_bimg.close()
fd_bedt.close()
fd_posenc.close()

sock.close()


