from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40
from model_dgcnn import DGCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import open3d as o3d
from FGSM import fgsm_attack
from Saliency import attack_saliency
from Pertubation import pertubation_point
from utils import plot_class_wise_scores
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import json
import wandb
from tqdm import tqdm

wandb.init(project="my-test-project", entity="ellight")
def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py outputs'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def pointcloud(new_points):
  for i in range(len(new_points)):
    pointcloud_vis = new_points[i].detach().cpu().numpy()
    pointcloud_vis = pointcloud_vis.reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_vis)
    o3d.io.write_point_cloud("dgcnn_pointcloud_"+str(i)+".ply",pcd)

def test(args, io):
    #test_dataset = ModelNet40(partition='test', num_points=args.num_points)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN_cls(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
   
    with open('convert.txt') as f:
      data = f.read()
    js = json.loads(data)
    values = []
    for i in js.items():
      values.append(i[1])
    b= dict(list(enumerate(values)))
    inv_classes = b
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    total_targets = torch.zeros((0))
    total_preds = torch.zeros((0))
    total_accu=0
    total_data_no=0
    criterion = nn.NLLLoss()
    #for data,label in enumerate(tqdm(test_loader, position=0, leave=False)):
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        #rint("first",data.shape)
        batch_size = data.size()[0]
        if args.attack_type=='fgsm':
          data = fgsm_attack(model, criterion, data, label, args.eps,'d')
        elif args.attack_type=='jsma':
          data = data.permute(0, 2, 1)
          data = attack_saliency(model, criterion, data, label, args.alpha, args.kl, args.num_drop,'d')
          data = data.permute(0, 2, 1)
        elif args.attack_type=='shift':
          data =  pertubation_point(data)
        
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        acc = torch.sum(preds == label) / preds.shape[0]
        total_accu += acc.item() * preds.shape[0]
        total_data_no += preds.shape[0]
        print("Batch wise Accuarcy {0:.4f}".format(acc.item()*100))
        total_preds = torch.cat([total_preds, preds.detach().cpu().squeeze()], dim=0)
        total_targets = torch.cat([total_targets, label.detach().cpu().squeeze()], dim=0)
    #test_true = np.concatenate(test_true)
    #test_pred = np.concatenate(test_pred)
    #test_acc = metrics.accuracy_score(test_true, test_pred)
    #avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    #outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    #io.cprint(outstr)
    total_targets = total_targets.detach().cpu().numpy()
    total_preds = total_preds.detach().cpu().numpy()
    confus_mat = confusion_matrix(total_targets, total_preds)
    recall_sco = recall_score(total_targets, total_preds, average=None)
    precision_sco = precision_score(total_targets, total_preds, average=None)
    io.cprint("Accuracy: {0:.4f}".format(total_accu *100 / total_data_no))
    plot_class_wise_scores(inv_classes, recall_sco, args.eps,args.attack_type,"recall_scores")
    plot_class_wise_scores(inv_classes, precision_sco, args.eps,args.attack_type, "precision_scores")
    data = [[label, val] for ((x,label), val) in zip(inv_classes.items(), recall_sco)]
    table = wandb.Table(data=data, columns = ["label", "value"])
    wandb.log({"1" : wandb.plot.bar(table, "label", "value",
                               title="Recall scores")})
    data2 = [[label, val] for ((x,label), val) in zip(inv_classes.items(), precision_sco)]
    table2 = wandb.Table(data=data2, columns = ["label", "value"])
    wandb.log({"2" : wandb.plot.bar(table2, "label", "value",
                               title="Precision scores")})

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--attack_type', type=str, default='fgsm', metavar='N',
                        help='type of attack')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='Epsilon value for fgsm')
    parser.add_argument('--alpha', type=int, default=1,
                        help='Alpha value for saliency map')
    parser.add_argument('--kl', type=int, default=5,
                        help='Number of times the points have to be dropped') 
    parser.add_argument('--num_drop', type=int, default=200,
                        help='number of points to drop')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
