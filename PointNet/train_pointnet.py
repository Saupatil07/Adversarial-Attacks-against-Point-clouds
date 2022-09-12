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
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    path = Path("ModelNet40")
    wandb.init(project="Pointnet_ModelNet40")
    train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

    test_dataset = PointCloudData(path, True, "train", train_transforms)
    model = PointNet()
    model = model.train()
    model = model.to(device)
    #model.load_state_dict(torch.load("./save.pth"))
    fgsm_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)
    criterion = nn.NLLLoss()
    inv_classes = {i: cat for cat, i in test_dataset.classes.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    continue_training = True
    
    if continue_training:
        model.load_state_dict(torch.load("./model.pth"))
        optimizer.load_state_dict(torch.load("./opt.pth"))

    total_loss = 0
    total_accu = 0
    total_data_no = 0
    #print("Attacking the pointnet using FGSM with epsilon ", epsilon)
    epoch = 20
    print(inv_classes)
    for j in range(epoch):
        print(j)
        loop = tqdm(enumerate(fgsm_dataloader), total=len(fgsm_dataloader), leave=False)
        for i, data in loop:
            labels = data['category']
            input_cloud = data['pointcloud']
            input_cloud = input_cloud.type(torch.FloatTensor)
            input_cloud = input_cloud.to(device)
            labels = labels.to(device)
            outputs, _, _ = model(input_cloud.permute(0, 2, 1))
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            total_data_no += labels.shape[0]
            pred = torch.argmax(outputs, dim=1)
            total_accu += torch.sum(pred == labels)
            loop.set_description(f'[{i}]/[{len(fgsm_dataloader)}], loss {round(total_loss/total_data_no, 2)}')
            wandb.log({"loss": round(total_loss/total_data_no, 2)})
        
        torch.save(model.state_dict(), "model.pth")
        torch.save(optimizer.state_dict(), "opt.pth")
        print(f'Epoch [{j}]/[{epoch}], loss: {total_loss / total_data_no}, accuracy: {total_accu / total_data_no}')
        wandb.log({"accuracy": total_accu/total_data_no})
        total_loss = 0
        total_data_no = 0
        total_accu = 0