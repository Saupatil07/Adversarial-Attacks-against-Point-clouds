
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

def fgsm_attack(model, criterion, point, labels, eps) :
    point.requires_grad = True
    #model = model.train()
    outputs, _, _= model(point.transpose(1,2))
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    attack_data = point + eps*point.grad.sign()
    return attack_data

def attack(model, criterion, point, label, eps, pointcloud_form=False):
  new_points = fgsm_attack(model, criterion, point, label, eps)
  model = model.eval()
  outputs, __, __ = model(new_points.transpose(1,2))
  _, preds = torch.max(outputs.data, 1)
  if (pointcloud_form):
    pointcloud_vis = new_points.detach().cpu().numpy()
    pointcloud_vis = pointcloud_vis.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_vis)
    o3d.io.write_point_cloud("after_attack_new.ply",pcd)

  return preds

if __name__ == '__main__':
    wandb.init(project="FGSM_Pointnet")
    parser = argparse.ArgumentParser(description="Performing fsgm on PointNet")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.00, help="Epsilon paramter for attacking the model")

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
    fgsm_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.NLLLoss()
    inv_classes = {i: cat for cat, i in test_dataset.classes.items()};
    print(inv_classes)
    

    
    epsilon_list = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] 

    for eps in epsilon_list:
      total_targets = torch.zeros((0))
      total_preds = torch.zeros((0))
      total_accu = 0
      total_data_no = 0
      print("Attacking the pointnet using FGSM with epsilon ", eps)
      loop = tqdm(enumerate(fgsm_dataloader),
                  total=len(fgsm_dataloader),
                  leave=False)
      for i, data in loop:
          labels = data['category']
          input_cloud = data['pointcloud']
          input_cloud = input_cloud.type(torch.FloatTensor)
          input_cloud = input_cloud.to(device)
          labels = labels.to(device)
          preds = attack(model, criterion, input_cloud, labels, eps=eps)
          
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
      wandb.log({"Accuracy vs Esilon": total_accu * 100 / total_data_no, "eps": eps})
      plot_class_wise_scores(inv_classes, recall_sco, "recall scores", eps)
      plot_class_wise_scores(inv_classes, precision_sco, "precision scores", eps)
