# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import math
 
# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)
 
# to run GUI event loop
plt.ion()
 
# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
data = np.random.rand(10, 10) * 2.0 - 1.0
plot = ax.imshow(data, cmap='turbo')
cbar = plt.colorbar(plot)
cbar.ax.set_autoscale_on(True)
 
# setting title
plt.title("Geeks For Geeks", fontsize=20)
 
# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
 
# Loop
for i in range(50):
    # creating new Y values
    data = np.random.rand(10, 10) * math.sin(i/10.0)
    plot.set_data(data)
    cbar.update_normal(plot)

    # drawing updated values
    figure.canvas.draw()
 
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()
 
    time.sleep(0.1)
