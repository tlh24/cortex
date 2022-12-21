import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

# Random data
data = np.random.rand(10, 10)

# Plot data
plot = ax.imshow(data)

# Create colorbar
cbar = plt.colorbar(plot)
cbar_ticks = np.linspace(0., 1., num=6, endpoint=True)
cbar.ax.set_autoscale_on(True)
cbar.set_ticks(cbar_ticks)

plt.show(block=False)
plt.draw()
fig.canvas.draw()
fig.canvas.flush_events()
time.sleep(3)

def update():
	new_data  = 2.*np.random.rand(10, 10)

	plot.set_data(new_data)
	cbar.set_clim(vmin=0,vmax=2)
	cbar_ticks = np.linspace(0., 2., num=11, endpoint=True)
	cbar.set_ticks(cbar_ticks) 
	cbar.draw_all() 
	plt.draw()

	plt.show()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(2)

update()
