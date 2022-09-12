import torch
def fgsm_attack(model, criterion, point, labels, eps, model_name) :
  point.requires_grad = True
  if model_name=='p':
    outputs,_,_= model(point.transpose(1,2))
  else:
    outputs= model(point)
  model.zero_grad()
  loss = criterion(outputs, labels)
  loss.backward()
  attack_data = point + eps*point.grad.sign()
  return attack_data
