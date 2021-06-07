import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random
import numpy as np
from PIL import Image
import pdb

# For proper experimentation, we should start with 
# something that has known intrinsic, low-dimensional structure
# and see if we can extract it as well as the convnet in paint.py
# which, perhaps surprisingly, works very well!  
# It's a fairly vanilla convolve and downsample architecture 
# I'm not sure how I came up with all this, must have copied from elsewhere, but pytorch sure is great. 
# now, the question is -- can we make a network that learns the same thing in an *unsupervised* manner? 

surf = tiledsurface.Surface()
blackfill = np.zeros((64, 64, 4), dtype=np.uint8)
blackfill[:,:,3] = 255 # fully opaque

def make_drawing(nlines, save_png=None, override=None):
    npoints = nlines + 1
    # draws to 'surf'. 
    color = np.random.rand(npoints,3)
    color = color / np.linalg.norm(color, axis=1, keepdims=True) 
        # saturate the colors
    color[0,:] = [0.0, 0.0, 0.0]
    dist = 0.0
    while dist < 20.0:
        location = np.random.rand(npoints,2) * 38 + 13
        dist = 0.0
        for k in range(nlines):
            dist = dist + np.linalg.norm(location[k,:] - location[k+1,:])
    pressure = np.random.rand(npoints) * 0.3 + 0.7
    radius = np.random.rand(npoints) * 5.0 + 1.5
    radius[0] = 2.0
    brush_num = np.random.randint(0, brush_count)
    # sort the points along the x-axis to help the decoder. 
    location = location[np.argsort(location[:,0])]
    if override is not None: 
        # vector in the same order as out, below.
        # first values can remain rand.. 
        # (this is to generate output from the analyzer network..)
        k = 0
        j = 3*nlines
        color[1:,:] = np.clip(np.reshape(override[k:j]/20.0, (nlines, 3)), 0.0, 1.0).astype(np.float64)
        k = j
        j = k + npoints*2
        location = np.reshape(override[k:j], (npoints,2)).astype(np.float64)
        k = j
        j = k + npoints
        pressure = np.clip(override[k:j]/20.0, 0.0, 1.0).astype(np.float64)
        k = j
        j = k + nlines
        radius[1:] = np.clip(override[k:j]/10.0, 0.5, 5.0).astype(np.float64)
        k = j
        j = k + brush_count
        brush_num = np.argmax(override[k:j])
    if False:
        npoints = 4
        # fixed override
        location = np.reshape([12.0,12.0, 12.0,12.0, 32.0,52.0, 52.0, 20.0], (npoints,2))
        color = np.reshape([1.0,0.0,0.0, 1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0], (npoints,3))
        pressure = [0.0, 1.0, 1.0, 1.0] # this is sorta like button-press...
        radius = [1.0, 1.0, 4.0, 5.0]
    # need to duplicate the first point, with pressure 0, a long time in the past, 
    # to keep from stroking from the previous internal state. 
    repeat = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    color = np.repeat(color, repeat[:npoints], axis=0)
    location = np.repeat(location, repeat[:npoints], axis=0)
    pressure = np.repeat(pressure, repeat[:npoints], axis=0)
    radius = np.repeat(radius, repeat[:npoints], axis=0)
    pressure[0] = 0.0
    # (color is applied before the stroke..)
    surf.load_from_numpy(blackfill, 0, 0)
    t0 = time()
    brsh[brush_num].new_stroke() # reset the internal variables.. 
    for i in range(npoints+1):
        if i == 0:
            dtime = 100.0 # long simulated pause = brush fully released.
        else:
            dtime = 0.1
        bi[brush_num].set_color_rgb(color[i,:])
        bi[brush_num].set_base_value('radius_logarithmic', math.log(radius[i]))
        surf.begin_atomic()
        brsh[brush_num].stroke_to(surf.backend, 
            location[i,0],location[i,1], 
            pressure[i],
            0.0, 0.0, # x and y tilt
            dtime, 1.0, 0.0, 0.0) 
            # above: time, view zoom, view rotation, barrel rotation
        surf.end_atomic()
    # endfor
    
    #return the parameters. 2 lines: three points, two colors
    one_hot = np.zeros(brush_count)
    one_hot[brush_num] = 1.0
    out = np.concatenate((color[2:,:]*20, 
                         location[1:,:], 
                         pressure[1:]*20, 
                         radius[2:]*10,
                         one_hot),
                         axis=None)
    if save_png is not None:
        length = np.linalg.norm(location[1,:] - location[2,:])
        print(f'file {save_png} \nparams {out} - length {length}')
        surf.save_as_png(save_png)
    return out

device = torch.device('cuda:1')
batch_size = 8 # 64 converges more slowly, obvi
niters = 500000
nlines = 1

params = make_drawing(nlines)
print(f'parameters size {params.size}')
params_size = params.size

batch_size = 1; 
y = torch.zeros((batch_size, params_size)).cuda(device)
x = torch.zeros((batch_size, 3, 32, 32)).cuda(device)

for k in range(niters):
    #generate the training data
    for i in range(batch_size):
        params = make_drawing(nlines)
        y[i,:] = torch.from_numpy(params)
        with surf.tile_request(0,0, readonly = True) as t1: 
            t2 = t1[16:48,16:48,0:3]
            im = torch.from_numpy(t2.astype(np.float32))
            im = torch.transpose(im, 0, 2) # transpose 'colors' to 'channels'
            im = im.contiguous()
            x[i,:,:,:] = im
            
    
    
# model sketch
input_size = 32*32*3 # 3072
nroute = 256
output_size = input_size / 6
wroute = torch.randn(input_size, nroute).cuda(device)
wroute = torch.mul(wroute,  sqrt(2 / input_size)) #will need to adjust activations here... 
wpg = torch.randn(input_size, output_size)
wpg = torch.mul(wpg, sqrt(2/input_size))
# now we need some way of controlling all the pass-gates..
# the naive approach is to have one pg control weight for each of the routing neurons
# but this becomes unweildy: input-output matrix of weights is 3MB; 
# with 256 routing neurons, this becomes 800 MB!  Nonsense. 
# the brain probably solves the problem via topology, or random selection of 
# 'which neuron controls which pass-gate' 
# the brain also has an extra dimension of dendritic processing here -- 
# pass-gates can silence various parts of various dendrites selectively, 
# and (perhaps importantly), the dendrites are arranged in a branching, hierarchical fashion
# (which mirrors the fractal world!)
# um, yeah, we should probably impress structure to the pass-gate control. 
# how? fourier transform? 
wroute_pg = torch.randn(
def model(x):
    xv = torch.reshape(x, (input_size)); # reshape into a vector.  destroy spatial info! 
    # first task is to generate the routing neurons ('pv')
    route = torch.matmul(xv, wroute)
    
