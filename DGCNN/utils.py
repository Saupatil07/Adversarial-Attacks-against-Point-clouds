import torchvision.transforms as transforms
from transforms import Normalize, PointSampler, ToTensor
import matplotlib.pyplot as plt
import numpy as np

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

def plot_class_wise_scores(class_names, scores, epsilon,attack,score_name):
    fig, ax = plt.subplots(1,1,figsize=(30,15))
    x_axis = np.arange(scores.shape[0])
    for i in range(scores.shape[0]):
          ax.bar([x_axis[i]], [scores[i]], label=class_names[i])
    ax.set_xlabel("class number")
    ax.set_ylabel(score_name)
    plt.legend()
    plt.savefig(score_name+"_"+attack+"_"+str(epsilon)+'.png')
    plt.show()

if __name__ == '__main__':
    pass
