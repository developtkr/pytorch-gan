import os
%matplotlib inline

import matplotlib.lines as mlines
import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchsummary import summary as summary_model

from model import pix2pix as p2pmodel

import logging
from datetime import datetime

run_code='test_p2p'
lr = 2e-4
batch_size=32
iters = 1000
workers=0
image_size=64
beta1=0.5
lambda_l1=100

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda=False
if torch.cuda.is_available():
    use_cuda=True

# transform
transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize([image_size,image_size]),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                           ])

# Pair data loader
class MyDataset(Dataset):
    def __init__(self, image1_paths, image2_paths, transform=None):
        self.image1_paths = image1_paths
        self.image2_paths = image2_paths
        self.transform = transform
        
    def __getitem__(self, index):
        img1 = Image.open(self.image1_paths[index])
        img2 = Image.open(self.image2_paths[index])
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2
    
    def __len__(self):
        return len(self.image1_paths)

#define data path 
image1_path = './train/input/'
image2_path = './train/target/'

test1_path = './test/input/'
test2_path = './test/target/'

node_list = os.listdir(image1_path)
sol_list = os.listdir(image2_path)

node_list_test = os.listdir(test1_path)
sol_list_test = os.listdir(test2_path)

node_list.sort()
sol_list.sort()

image1_paths = list(map(lambda x:image1_path+x, node_list))
image2_paths = list(map(lambda x:image2_path+x, sol_list))

test1_paths = list(map(lambda x:test1_path+x, node_list_test))
test2_paths = list(map(lambda x:test2_path+x, sol_list_test))


dataset = MyDataset(
    image1_paths, image2_paths, transform=transform)

testset = MyDataset(
    test1_paths, test2_paths, transform=transform)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                                         shuffle=True, num_workers=workers)

# loss 
criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

gen = p2pmodel.G(n_channel_input=1, n_channel_output=1, n_filters=16).to(device)
dis = p2pmodel.D(n_channel_input=1, n_channel_output=1, n_filters=16).to(device)
gen.apply(weights_init)
dis.apply(weights_init)

summary_model(gen, input_size=(3,64,64))
summary_model(dis, input_size=(6,64,64))

model_dir='./models/'+run_code+'/'
G_save_path = model_dir + 'generator.pth'
D_save_path = model_dir + 'critic.pth'
G_total_save_path = model_dir + 'generator_total.pth'
D_total_save_path = model_dir + 'critic_total.pth'

if not os.path.isdir(model_dir): os.makedirs(model_dir)

if os.path.exists(G_total_save_path):
    gen = torch.load(G_total_save_path)

if os.path.exists(D_total_save_path):
    dis = torch.load(D_total_save_path)

real_label = 1
fake_label = 0

# plt.rcParams["figure.figsize"] = (8, 6)

label = torch.FloatTensor(batch_size)
label = label.cuda()
label = Variable(label)

s = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
logging.info(s+ ' Learning Start..')
print(s+ ' Learning Start..')

for i in range(iters): 
    real_batch = next(iter(dataloader))
    inputs = real_batch[0].to(device)
    targets = real_batch[1].to(device)
     
    # Train D
    dis.zero_grad()
    # train - real
    real_input = torch.cat((inputs, targets), 1)
    out_d = dis(real_input)
    label.data.resize_(out_d.size()).fill_(real_label)
    err_real = criterion(out_d, label)
    err_real.backward()
    
    d_x_y = out_d.data.mean()

    
    # train - fake
    fake_target = gen(inputs)
    fake_input = torch.cat((inputs, fake_target.detach()), 1)
    out_d_fake = dis(fake_input)
    label.data.resize_(out_d.size()).fill_(fake_label)
    err_fake = criterion(out_d_fake, label)
    err_fake.backward()
    d_x_gx = out_d_fake.data.mean()
    
    err_d = (err_real + err_fake) * 0.5
    optimizerD.step()
    
        
    # Train G
    gen.zero_grad()
    gen_input = torch.cat((inputs, fake_target), 1)
    out_g_fake = dis(gen_input)
    label.data.resize_(out_d.size()).fill_(real_label)
    
    err_gen = criterion(out_g_fake, label)
    err_l1_gen = criterion_l1(fake_target, targets)
    
    err_g = err_gen+ err_l1_gen*lambda_l1
    err_g.backward()
    d_x_gx_2 = out_g_fake.data.mean()
    
    optimizerG.step()
    
    if i%100==0:
        s = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        print (s + '=> iters [{}]: Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                i,
                err_d,
                err_g,
                d_x_y,
                d_x_gx,
                d_x_gx_2,
                ))
        logging.info(s+' iters [{}]: Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                i,
                err_d,
                err_g,
                d_x_y,
                d_x_gx,
                d_x_gx_2,
                ))
        
        summary.add_scalar('loss/Loss_D', err_d, i)
        summary.add_scalar('loss/Loss_G', err_g, i)
        summary.add_scalar('loss/D(x)', d_x_y, i)
        summary.add_scalar('loss/D(G(z))', d_x_gx, i)
        summary.add_scalar('loss/D(G(z))2', d_x_gx_2, i)
        
        # Add Image (test)
        testImg = getTestImages(gen)
        summary.add_image('TestImage', testImg, i, dataformats='HW')
        
        # Add Graph
        summary.add_graph(gen, node_batch, False)
        summary2.add_graph(dis, fake_input, False)

        torch.save(gen.state_dict(), G_save_path)
        torch.save(dis.state_dict(), D_save_path)
        torch.save(gen, G_total_save_path)
        torch.save(dis, D_total_save_path)