import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tt
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

if __name__ == '__main__':
    path = Path("ModelNet10")
    train_transforms = tt.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

    test_dataset = PointCloudData(path, True, "test", train_transforms)
    model = PointNet()
    #model = model.eval()
    model = model.to(device)
    if(model.load_state_dict(torch.load("./save.pth"))):
      print("Loaded pretrained weights")
    fgsm_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.NLLLoss()

    for i, data in enumerate(tqdm(fgsm_dataloader, position=0, leave=False)):
        labels = data['category']
        input_cloud = data['pointcloud']
        input_cloud = input_cloud.type(torch.FloatTensor)
        input_cloud = input_cloud.to(device)
        labels = labels.to(device)
        preds, _, _ = model(input_cloud.permute((0, 2, 1)))
        print(torch.argmax(preds, dim=1), labels)
        