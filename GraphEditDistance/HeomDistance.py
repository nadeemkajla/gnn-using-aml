#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    HeomDistance.py: Computes an Heterogeneous Euclidean Overlap Metric (HEOM) distance

    * Bibliography: Salim Jouili et al. (2009) "Attributed Graph Matching using Local Descriptions"

    Usage:
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

import numpy as np

import pdb

# Own modules


__author__ = "Nadeem Kajla"
__email__ = "nadeem.kajla@gmail.com"


class Heomd(nn.Module):

    # Constructor
    def __init__(self, args={}):
        super(Heomd, self).__init__()
        self.args = args

    def forward(self, v1, am1, sz1, v2, am2, sz2):
        byy = v2.unsqueeze(1).expand((v2.size(0), v1.size(1), v2.size(1), v2.size(2))).transpose(1, 2)
        bxx = v1.unsqueeze(1).expand_as(byy)

        bdxy = torch.sqrt(torch.sum((bxx - byy) ** 2, 3))

        # Create a mask for nodes
        node_mask2 = torch.arange(0, bdxy.size(1)).unsqueeze(0).unsqueeze(-1).expand(bdxy.size(0),
                                                                                     bdxy.size(1),
                                                                                     bdxy.size(2)).long()
        node_mask1 = torch.arange(0, bdxy.size(2)).unsqueeze(0).unsqueeze(0).expand(bdxy.size(0),
                                                                                    bdxy.size(1),
                                                                                    bdxy.size(2)).long()

        if v1.is_cuda:
            node_mask1 = node_mask1.cuda()
            node_mask2 = node_mask2.cuda()
        node_mask1 = Variable(node_mask1, requires_grad=False)
        node_mask2 = Variable(node_mask2, requires_grad=False)
        node_mask1 = (node_mask1 >= sz1.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask1))
        node_mask2 = (node_mask2 >= sz2.unsqueeze(-1).unsqueeze(-1).expand_as(node_mask2))

        node_mask = node_mask1 | node_mask2

        maximum = bdxy.max()

        bdxy.masked_fill_(node_mask, float(maximum))

        bm1, _ = bdxy.min(dim=2)
        bm2, _ = bdxy.min(dim=1)

        bm1.masked_fill_(node_mask.prod(dim=2), 0)
        bm2.masked_fill_(node_mask.prod(dim=1), 0)
        d = bm1.sum(dim=1) + bm2.sum(dim=1)
       # d = self.heom(bm1.data.numpy(), bm2.data.numpy(), step=bm1.size()[1])
       #  d = Variable(torch.FloatTensor(d), requires_grad=True);
        # print(d.size())
        return d / (sz1.float() + sz2.float())

    # d = bm1.sum(dim=1) + bm2.sum(dim=1)
    # # d = [375,1]
    # # print(d)
    # return d / (sz1.float() + sz2.float())



    # def get_ranges(self, arr1, arr2):

    def heom(self, arr1, arr2, step=8):
        dist = [];
        for j in range(len(arr1)):
            maximum = int(0)
            d = float(0)
            Dist = float(0)
            v1 = arr1[j]
            v2 = arr2[j]
            if type(v1) is np.ndarray:
                v1 = v1.tolist()
            if type(v2) is np.ndarray:
                v2 = v2.tolist()

            if (len(v1) < len(v2)):
                maximum = len(v2);
                for ind, val in enumerate(v2, start=len(v1)):
                    v1.insert(ind, None)
            else:
                maximum = len(v1)
                for ind, val in enumerate(v1, start=len(v2)):
                    v2.insert(ind, None)

            for i in range(maximum):
                if (v1[i] == None or v2[i] == None):
                    d = 1
                elif (isinstance(v1[i], str)):
                    if (not (str(v2[i]) == str(v1[i]))):
                        d = 1
                elif (isinstance(v1[i], int)):
                    d = float(abs((int(v1[i]) - int(v2[i])))) / 20
                elif (isinstance(v1[i], float)):
                    d = float(abs((float(v1[i]) - float(v2[i])))) / 20
                Dist += d * d
            dist.append(np.sqrt(Dist))

        return dist
