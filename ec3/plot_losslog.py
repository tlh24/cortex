import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt
import time



plot_rows = 2
plot_cols = 4
figsize = (18, 10)
plt.ion()
fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False

while True: 
	with open("loss_log.txt", 'r') as x:
		data = list(csv.reader(x, delimiter="\t"))

	data = np.array(data)
	data = data.astype(float)

	ax[0,0].cla()
	ax[0,0].plot(data[:,0], np.log(data[:, 1]), 'b')
	ax[0,0].set(xlabel='iteration')
	ax[0,0].set_title('log loss')

	labels = ["vit","vit_to_prt","encoder","vxpx","prt","prt_to_edit"]
	for i in range(6): 
		lab = labels[i]
		r = (i+1) // 4
		c = (i+1) % 4
		ax[r,c].cla()
		ax[r,c].plot(data[:,0], data[:,i+2], 'b')
		ax[r,c].set_title(f'st.dev {lab} output')

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(0.2)
	print("tick")
# plt.show()
