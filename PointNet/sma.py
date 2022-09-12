
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transforms import Normalize, PointSampler, RandomNoise, RandRotation_z, ToTensor
from dataset import PointCloudData
from model import PointNet
import open3d as o3d
from path import Path
import argparse
import sklearn
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from tqdm import tqdm
from utils import plot_class_wise_scores
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
      #if i % (num_rounds // 5) == 0:
          #print('Iteration {}/{}, success {}/{}\n'
          #      'Point num: {}/{}'.
          #      format(i, num_rounds, success_num, B,
          #              K, K_))

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

if __name__ == '__main__':
    wandb.init(project="SMA_PointNet")
    parser = argparse.ArgumentParser(description="Performing Saliency Map Attack on PointNet")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--num_drop", type=int, default=50, help="num_drops paramter for attacking the model")

    args = parser.parse_args()

    path = Path("ModelNet40")
    train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

    test_dataset = PointCloudData(path, True, "test")
    model = PointNet(classes=40)
    model = model.eval()
    model = model.to(device)
    model.load_state_dict(torch.load("./model.pth"))
    sma_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.NLLLoss()
    inv_classes = {i: cat for cat, i in test_dataset.classes.items()};
    print(inv_classes)

    alpha = 1
    kl = 5
    num_drop_list = [50,100,150,200,250,300]
    #num_drop_list = [args.num_drop]

    for num_drop in num_drop_list:
      total_targets = torch.zeros((0))
      total_preds = torch.zeros((0))
      total_accu = 0
      total_data_no = 0
      print("Attacking the pointnet using SMA with num_drops ", num_drop)
      loop = tqdm(enumerate(sma_dataloader),
                  total=len(sma_dataloader),
                  leave=False)
      for i, data in loop:
          labels = data['category']
          input_cloud = data['pointcloud']
          input_cloud = input_cloud.type(torch.FloatTensor)
          input_cloud = input_cloud.to(device)
          labels = labels.to(device)
          new_data = attack_saliency(model, criterion, input_cloud, labels, alpha, kl, num_drop, 'f')
          outputs,_,_ = model(new_data.transpose(1,2))
          _, preds = torch.max(outputs.data,1)
          acc = torch.sum(preds == labels) / preds.shape[0]
          total_accu += acc.item() * preds.shape[0]
          total_data_no += preds.shape[0]
          #print("Batch wise Accuarcy {0:.4f}".format(acc.item()*100))
          loop.set_description(f'Batch wise accuracy {acc.item()}')
          total_preds = torch.cat([total_preds, preds.detach().cpu().squeeze()], dim=0)
          total_targets = torch.cat([total_targets, labels.detach().cpu().squeeze()], dim=0)

      total_targets = total_targets.detach().cpu().numpy()
      total_preds = total_preds.detach().cpu().numpy()
      confus_mat = confusion_matrix(total_targets, total_preds)
      recall_sco = recall_score(total_targets, total_preds, average=None)
      precision_sco = precision_score(total_targets, total_preds, average=None)
      print("Accuracy: {0:.4f}".format(total_accu *100 / total_data_no))
      wandb.log({"Accuracy vs Num_drops": total_accu * 100 / total_data_no, "num_drop": num_drop})
      plot_class_wise_scores(inv_classes, recall_sco,num_drop,'sma', "recall scores")
      plot_class_wise_scores(inv_classes, precision_sco, num_drop, 'sma',"precision scores")
