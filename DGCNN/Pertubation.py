# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:36:53 2022

@author: ssp2
"""
import torch
import torch.nn as nn

def pertubation_point(point_cloud):
    x,y,z = point_cloud.shape
    epsilon = torch.empty(x,y,z).to('cuda')
    epsilon = nn.init.xavier_normal_(epsilon)
    pert_pc = torch.add(point_cloud,epsilon)
    return pert_pc
