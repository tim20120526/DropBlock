import torch
import numpy as np
import torch.nn.functional as F

def Dropblock(feature_input,block_size,keep_prob):  
    b,c,w,h = feature_input.size()
    bernoulli_mask_line = w - (block_size-1)
    whole_mask = torch.ones(w,h).cuda()

    ganma = ((1-keep_prob)/(block_size**2))*(w**2/(w-block_size+1)**2)

    coorx = (block_size-1)//2
    coory = (block_size-1)//2+bernoulli_mask_line
    whole_mask[coorx:coory,coorx:coory].bernoulli_(1-ganma)

    f = torch.le(whole_mask,0.5)
    zero_position = torch.nonzero(f)
    for coor in zero_position:
        x1 = coor[1] - (block_size-1)/2
        y1 = coor[0] - (block_size-1)/2
        x2 = coor[1] + (block_size-1)/2 +1
        y2 = coor[0] + (block_size-1)/2 +1
        whole_mask[y1:y2,x1:x2] = 0

    count_mask = w*h
    count_ones = whole_mask.sum()

    whole_mask = whole_mask.unsqueeze(0).unsqueeze(0)
    whole_mask = whole_mask.expand(b,c,w,h)
    feature_output = feature_input* whole_mask
    feature_output = feature_output* (count_mask/count_ones)

    return feature_output








