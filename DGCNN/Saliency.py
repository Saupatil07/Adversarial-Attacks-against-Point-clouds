# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:35:48 2022

@author: ssp2
"""

import numpy as np
import torch

def get_gradient_saliency(data, target, criterion, model,model_name):
    data.requires_grad_()
    # forward pass
    if model_name=='f':
      logits,_,_ = model(data)
    else:
      logits = model(data)
    loss = criterion(logits, target)
    loss.backward()
    with torch.no_grad():
        grad = data.grad.detach()  # [B, 3, K]
        pred = torch.argmax(logits, dim=-1)  # [B]
        num = (pred == target).sum().detach().cpu().item()
    return grad, num

def attack_saliency(model, criterion, data, target, alpha, k, num_drop,model_name):
  B, K_ = data.shape[:2]
  data = data.float().cuda().detach()
  data = data.transpose(1, 2).contiguous()  # [B, 3, K]
  target = target.long().cuda().detach()  # [B]
  num_rounds = int(np.ceil(float(num_drop) / float(k)))
  for i in range(num_rounds):
      K = data.shape[2]

      # number of points to drop in this round
      k = min(k, num_drop - i * k)

      # calculate gradient of loss
      grad, success_num = \
          get_gradient_saliency(data, target, criterion, model,model_name)  # [B, 3, K]
      if i % (num_rounds // 5) == 0:
          print('Iteration {}/{}, success {}/{}\n'
                'Point num: {}/{}'.
                format(i, num_rounds, success_num, B,
                        K, K_))

      with torch.no_grad():
          # compute center point
          center = torch.median(data, dim=-1)[0].\
              clone().detach()  # [B, 3]

          # compute r_i as l2 distance
          r = torch.sum((data - center[:, :, None]) ** 2,
                        dim=1) ** 0.5  # [B, K]

          # compute saliency score
          saliency = -1. * (r ** alpha) * \
              torch.sum((data - center[:, :, None]) * grad,
                        dim=1)  # [B, K]

          # drop points with highest saliency scores w.r.t. gt labels
          # note that this is for untarget attack!
          _, idx = (-saliency).topk(k=K - k, dim=-1)  # [B, K - k]
          data = torch.stack([
              data[j, :, idx[j]] for j in range(B)
          ], dim=0)  # [B, 3, K - k]

  # end of dropping
  # now data is [B, 3, K - self.num_drop]
  return data.permute(0,2,1) # Permute to reshape data 
