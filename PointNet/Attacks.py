#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fgsm_attack(model, criterion, point, labels, eps) :
    point.requires_grad = True
    outputs, _, _= model(point.transpose(1,2))
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    attack_data = point + eps*point.grad.sign()
    return attack_data

def pertubation_point(point_cloud):
    x,y,z = point_cloud.shape
    epsilon = torch.empty(x,y,z)
    epsilon = nn.init.xavier_normal_(epsilon)
    pert_pc = torch.add(point_cloud,epsilon)
    return pert_pc

def attack(model, criterion, point, label, eps, k,attack_type, pointcloud_form=False):
    if attack_type =='fgsm':
        
        new_points = fgsm_attack(model, criterion, point, label, eps)
    else:
        new_points = pertubation_point(point)
    
    LL = labels.detach().cpu().numpy()[0]
    outputs, __, __ = model(new_points.transpose(1,2))
    _, preds = torch.max(outputs.data, 1)
    if (pointcloud_form):
        for i in range(len(point)):
            pointcloud_vis = new_points[i].detach().cpu().numpy()
            pointcloud_vis = pointcloud_vis.reshape(-1,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud_vis)
            o3d.io.write_point_cloud("before_attack_new_pert_point_"+str(i)+"_"+str(k)+str(LL)+".ply",pcd)

    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performing fsgm on PointNet")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon paramter for attacking the model")

    args = parser.parse_args()

    path = Path("ModelNet10")
    train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

    test_dataset = PointCloudData(path, True, "test", train_transforms)
    model = PointNet()
    model = model.train()
    model = model.to(device)
    model.load_state_dict(torch.load("./save.pth"))
    fgsm_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.NLLLoss()
    inv_classes = {i: cat for cat, i in test_dataset.classes.items()};

    total_targets = torch.zeros((0))
    total_preds = torch.zeros((0))
    total_accu = 0
    total_data_no = 0

    print("Attacking the pointnet using FGSM with epsilon ", args.epsilon)
    for i, data in enumerate(tqdm(fgsm_dataloader, position=0, leave=False)):
        labels = data['category']
        input_cloud = data['pointcloud']
        input_cloud = input_cloud.type(torch.FloatTensor)
        input_cloud = input_cloud.to(device)
        labels = labels.to(device)
        torch.cuda.empty_cache()
        preds = attack(model, criterion, input_cloud, labels,eps=1e-3,k=i,attack_type='fgsm')
        
        acc = torch.sum(preds == labels) / preds.shape[0]
        total_accu += acc.item() * preds.shape[0]
        total_data_no += preds.shape[0]
        print("Batch wise Accuarcy {0:.4f}".format(acc.item()*100))
        total_preds = torch.cat([total_preds, preds.detach().cpu().squeeze()], dim=0)
        total_targets = torch.cat([total_targets, labels.detach().cpu().squeeze()], dim=0)
    
    total_targets = total_targets.detach().cpu().numpy()
    total_preds = total_preds.detach().cpu().numpy()
    confus_mat = confusion_matrix(total_targets, total_preds)
    recall_sco = recall_score(total_targets, total_preds, average=None)
    precision_sco = precision_score(total_targets, total_preds, average=None)
    print("Accuracy: {0:.4f}".format(total_accu *100 / total_data_no))
    plot_class_wise_scores(inv_classes, recall_sco, "recall scores")
    plot_class_wise_scores(inv_classes, precision_sco, "precision scores")

