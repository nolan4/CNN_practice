import torch
import torchvision
from torchvision import utils
# from basic_fcn import *
# from dataloader import *
# from utils import *
# import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import torch.nn
import matplotlib.pyplot as plt



batch_size_train = 64
batch_size_test = 1000
# assemble the train, test, and val sets
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./train_data', train=True, download=False, 
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=batch_size_train, shuffle=True)

teval_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./teval_data', train=False, download=False, 
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=batch_size_test, shuffle=True)




