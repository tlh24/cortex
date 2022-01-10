# Imports:

from __future__ import division, print_function
from time import time
from os.path import join
from itertools import product
import unittest
import sys
import os
import tempfile
import shutil

import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
import pdb

import torch

from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

home_dir = os.path.expanduser("~")
lib_dir = os.path.join(home_dir, "Dropbox/work/ML/mypaint/")
sys.path.append(lib_dir)
lib_dir = os.path.join(home_dir, "Dropbox/work/ML/mypaint/lib/")
sys.path.append(lib_dir)
print(lib_dir)
so_dir = os.path.join(home_dir, "Dropbox/work/ML/mypaint/build/lib.linux-x86_64-3.8/lib/")
sys.path.append(so_dir)

import mypaint
import tiledsurface 
import brush
import document

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        # print("compute_weight ", weight.size())

        return weight * sqrt(2 / fan_in)
    
    # Idea: rather than changing the intialization of the weights, 
    # here they just use N(0, 1), and dynamically scale the weights by 
    # sqrt(2 / fan_in), per He 2015. 
    # updates are applied to the unscaled weights (checked), 
    # which means that the gradient updates are also scaled by sqrt(2/fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
    

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
            nn.AvgPool2d(2), # downsample.
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out
    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# ------- #

surf = tiledsurface.Surface()
color = np.random.rand(2,3)
color = color / np.linalg.norm(color, axis=1, keepdims=True) # saturate the colors
location = np.random.rand(2,2)*38 + 13 # keep within one tile
pressure = np.random.rand(2) * 0.2 + 0.8
# total number of parameters: (3 + 2 + 1) * 2 = 12
blackfill = np.zeros((64, 64, 4), dtype=np.uint8)
blackfill[:,:,3] = 255 # fully opaque
#t0 = time()
#for k in range(2):
    #surf.load_from_numpy(blackfill, 0, 0)
    #surf.begin_atomic()
    #for i in range(2):
        #surf.draw_dab(location[i,0], location[i,1], 12, \
            #color[i,0],color[i,1],color[i,2], pressure[i], 1.0)
        ## x, y, radius, r, g, b, opacity, hardness
    #surf.end_atomic()
#print('%0.4fs, ' % (time() - t0,))
## plenty fast -- about 270us/frame with two dabs.
#surf.save_as_png('test_directPaint.png')

brushes_path = '/home/tlh24/Dropbox/work/ML/mypaint-brushes/brushes/'
brush_list = ['ramon/Round_Bl.myb',
              'ramon/Sketch_1.myb',
              'classic/charcoal.myb',
              'classic/calligraphy.myb', 
              'classic/dry_brush.myb']
brsh = []
bi = []
for i, brush_name in enumerate(brush_list):
    myb_path = os.path.join(brushes_path, brush_name)
    with open(myb_path, "r") as fp:
        bi.append(brush.BrushInfo(fp.read()))
    brsh.append(brush.Brush(bi[i]))
    
brush_count = len(brush_list)

#def make_drawing_1():
    #color = np.random.rand(2,3)
    #color = color / np.linalg.norm(color, axis=1, keepdims=True) 
        ## saturate the colors
    #location = np.random.rand(2,2)*38 + 13 # keep within one tile
    #pressure = np.random.rand(2) * 0.4 + 0.6
        ## total number of parameters: (3 + 2 + 1) * 2 = 12
    #surf.load_from_numpy(blackfill, 0, 0)
    #surf.begin_atomic()
    #for i in range(2):
        #surf.draw_dab(location[i,0], location[i,1], 12, \
            #color[i,0],color[i,1],color[i,2], pressure[i], 1.0)
        ## x, y, radius, r, g, b, opacity, hardness
    #surf.end_atomic()
    #arr = np.concatenate((color,(location-13.0)/38.0, pressure), axis=None)

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


# time to plug in the convnet. 
device = torch.device('cuda:0')
batch_size = 8 # 64 converges more slowly, obvi
niters = 500000
nlines = 1

params = make_drawing(nlines)
print(f'parameters size {params.size}')
params_size = params.size
#for i in range(20):
    #params = make_drawing(nlines, save_png=f'test_paint{i}.png')
    #make_drawing(nlines, override=params, save_png=f'test_paint{i}_check.png')
#params = np.array([ 8.41980934,11.27547026,13.4562242,19.33030319,39.35288239,18.90433121, 39.75945663, 17.75170565, 17.71341681, 33.08949709])
#make_drawing(nlines, override=params, save_png='test_paint_fail.png')
#quit()

analyzer = nn.Sequential(
    EqualConv2d(3, 16, 1), # RGB to channel
    nn.LeakyReLU(0.2), 
    ConvBlock(16, 32, 3, 1), # 32
    ConvBlock(32, 64, 3, 1), # 16
    ConvBlock(64, 128, 3, 1), # 8
    Reshape(batch_size, 128*64),
    EqualLinear(128*64, 512), 
    nn.LeakyReLU(0.2), 
    EqualLinear(512, params_size)).cuda(device)
requires_grad(analyzer, True)

optimizer = optim.Adam(analyzer.parameters(), lr=0.001, betas=(0.0, 0.99))
lossfunc = torch.nn.SmoothL1Loss() # mean reduction
lossfunc_noreduce = torch.nn.SmoothL1Loss(reduce=False) # no reduction along batch dim

y = torch.zeros((batch_size, params_size)).cuda(device)
x = torch.zeros((batch_size, 3, 64, 64)).cuda(device)
blackfill = np.zeros((64, 64, 4), dtype=np.uint8)
blackfill[:,:,3] = 255 # fully opaque
slowloss = 0.0
losses = np.zeros((2,niters))

for k in range(niters):
    #generate the training data
    for i in range(batch_size):
        params = make_drawing(nlines)
        ten = torch.from_numpy(params)
        y[i,:] = ten
        with surf.tile_request(0,0, readonly = True) as t1: 
            t2 = t1[:,:,0:3]
            im = torch.from_numpy(t2.astype(np.float32))
            im = torch.transpose(im, 0, 2) # transpose 'colors' to 'channels'
            im = im.contiguous()
            x[i,:,:,:] = im

    analyzer.zero_grad()
    predict = analyzer(x)
    # the loss depends on ordering (!!) permute that. 
    # this didn't work!  collapsed to zero length segments.
    #loss1 = lossfunc_noreduce(y, predict).sum(1) # predict is batch_size by param_size
    #permute = torch.tensor([0,1,2,5,6,3,4,8,7,9]) # this is for one line
    #loss2 = lossfunc_noreduce(y, predict[:,permute]).sum(1) #sum-to, not sum-from
    #loss2zero = (loss1 < loss2).type(torch.long)
    #loss1zero = 1 - loss2zero
    ## permuting causes the optimization to prefer very short segments. 
    #predlen = torch.norm(predict[3:5] - predict[5:7])
    #ylen = torch.norm(y[3:5] - y[5:7])
    #loss = (loss1*loss1zero + loss2*loss2zero).sum(0) + lossfunc(predlen, ylen)*10.0
    loss = lossfunc(y, predict)
    loss.backward()
    optimizer.step()
    slowloss = 0.99*slowloss + 0.01 * loss
    if k % 10 == 0 : 
        print(f'loss: {loss}; slowloss {slowloss}')
    losses[0,k] = loss
    losses[1,k] = slowloss
    
# plot it
if False:
    t = np.arange(0.0, niters, 1.0)
    plt.figure(1)
    plt.title('loss vs iteration')
    plt.plot(t, losses[0,:], 'b')
    plt.plot(t, losses[1,:], 'r')
    plt.show()

# run some trials
for i in range(batch_size):
    params = make_drawing(nlines)
    ten = torch.from_numpy(params)
    y[i,:] = ten
    with surf.tile_request(0,0, readonly = True) as t1: 
        t2 = t1[:,:,0:3]
        im = torch.from_numpy(t2.astype(np.float32))
        im = torch.transpose(im, 0, 2) # transpose 'colors' to 'channels'
        im = im.contiguous()
        x[i,:,:,:] = im
        
predict = analyzer(x)
override = predict.detach().cpu().numpy()
orig = y.detach().cpu().numpy()
for i in range(batch_size):
    make_drawing(nlines, override=orig[i,:], save_png=f'm{i}_original.png')
    make_drawing(nlines, override=override[i,:], save_png=f'm{i}_predict.png')
    print('\n')
