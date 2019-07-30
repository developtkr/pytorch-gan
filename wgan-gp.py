from models import resnet

import torch
import torch.optim as optim
from torchsummary import summary

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt

import os, time
import logging

# Model Run Code
run_code = 'wgan-gp_resnet32'

# Define Optimizer
lr = 1e-4
beta1=.5
beta2=.9
batch_size = 64

num_epochs = 20

# noise length
nz = 128

n_critic = 5

image_size=32
# Number of workers for dataloader
workers = 4

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Print Model Structure.
print_model = False

# Logging..
log_dir='../logs/' + run_code + '/'
model_dir='../logs/' + run_code + '/model/'

# for Log
if not os.path.isdir(log_dir): os.makedirs(log_dir)
logging.basicConfig(filename=log_dir+'logging.log', level=logging.INFO)

# for Model Save
if not os.path.isdir(model_dir): os.makedirs(model_dir)
    
G_save_path = model_dir + 'generator.pth'
D_save_path = model_dir + 'critic.pth'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

netG = resnet.GeneratorResNet32(resnet.ResBlock).to(device)
netD = resnet.DiscriminatorResNet32(resnet.ResBlock).to(device)

optimizerD = optim.Adam(resnetD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(resnetG.parameters(), lr=lr, betas=(beta1, beta2))

if print_model:
    summary(resnetG, input_size=(128, 1, 1))
    summary(resnetD, input_size=(3, 32, 32))


dataroot = "../data/"

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Gradient Penalty..
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

# fixed noise for test
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
GP = []

iters = 0


print("Starting Training Loop...")
logging.debug('Starting Training Loop...')

# For each epoch
start_time = time.time()
for iters in range(num_epochs * len(dataloader)):
    # For each batch in the dataloader
    ############################
    # Train D
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
        
    for i in range(n_critic):
        real_data = next(iter(dataloader))[0]
        netD.zero_grad()
        
        b_size = real_data.size(0)
        
        # train with real 
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = Variable(real_data)
        
        D_real = netD(real_data_v)
        D_real = D_real.mean()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
            
        with torch.no_grad():
            noisev = Variable(noise)  # totally freeze netG
        fake = Variable(netG(noisev).data)
        inputv = fake
        
        D_fake = netD(inputv)
        D_fake = D_fake.mean()

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        D_cost = D_fake - D_real 
        GP_cost = gradient_penalty
        
        c_loss = D_cost + GP_cost
        c_loss.backward()
        optimizerD.step()

        D_losses.append(D_cost.cpu().data.numpy())
        GP.append(GP_cost.cpu().data.numpy())
        
    ############################
    # Train G
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()
    
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    noisev = autograd.Variable(noise)
    
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G_cost = -G
    
    G_cost.backward()
    optimizerG.step()
    
    G_losses.append(G_cost.cpu().data.numpy())
    
    # Output training stats
    if iters % 10 == 0:
        iter_time = time.time() - start_time  
        start_time = time.time()
        print('[%d/%d]\t GP: %.4f\t D: %.4f\t G: %.4f\t %.4f sec'
              % (iters, num_epochs * len(dataloader), GP[-1], D_losses[-1], G_losses[-1], iter_time ))
        logging.info('[%d/%d]\t GP: %.4f\t D: %.4f\t G: %.4f\t %.4f sec'
              % (iters, num_epochs * len(dataloader), GP[-1], D_losses[-1], G_losses[-1], iter_time ))

        torch.save(netG.state_dict(), G_save_path)
        torch.save(netD.state_dict(), D_save_path)

    # Check how the generator is doing by saving G's output on fixed_noise
    if (iters % 10 == 0):
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    iters += 1
