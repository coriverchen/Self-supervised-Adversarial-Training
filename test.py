# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../tmp'))
	print(os.getcwd())
except:
	pass

#%%
import os
import argparse

import torch

import mixed_precision
from stats import StatTracker
from datasets2 import Dataset, build_dataset, get_dataset, get_encoder_size
from model_grad import Model
from checkpoint import Checkpoint
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers


parser = argparse.ArgumentParser(description='Infomax Representations -- Self-Supervised Training')
parser.add_argument("--verbosity", help="increase output verbosity")
# parameters for general training stuff
parser.add_argument('--dataset', type=str, default='STL10')
parser.add_argument('--batch_size', type=int, default=200,
                    help='input batch size (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')

# parameters for model and training objective
parser.add_argument('--classifiers', action='store_true', default=False,
                    help="Wether to run self-supervised encoder or"
                    "classifier training task")
parser.add_argument('--ndf', type=int, default=64,
                    help='feature width for encoder')
parser.add_argument('--n_rkhs', type=int, default=1024,
                    help='number of dimensions in fake RKHS embeddings')
parser.add_argument('--tclip', type=float, default=20.0,
                    help='soft clipping range for NCE scores')
parser.add_argument('--n_depth', type=int, default=8)
parser.add_argument('--use_bn', type=int, default=1)

# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./runs',
                    help='directory where tensorboard events and checkpoints will be stored')
parser.add_argument('--input_dir', type=str, default='/root/data/ILSVRC/Data/CLS-LOC/',
                    help="Input directory for the dataset. Not needed For C10,"
                    " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--cpt_load_path', type=str, default='abc.xyz',
                    help='path from which to load checkpoint (if available)')
parser.add_argument('--cpt_name', type=str, default='cifar_amdim_cpt.pth',
                    help='name to use for storing checkpoints during training')
parser.add_argument('--run_name', type=str, default='cifar_default_run',
                    help='name to use for the tensorbaord summary for this run')
# ...
args = parser.parse_args(args=[])

# create target output dir if it doesn't exist yet
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

# enable mixed-precision computation if desired
if args.amp:
    mixed_precision.enable_mixed_precision()

# set the RNG seeds (probably more hidden elsewhere...)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# get the dataset
dataset = get_dataset(args.dataset)
enc_size = get_encoder_size(dataset)

# get a helper object for tensorboard logging
log_dir = os.path.join(args.output_dir, args.run_name)
stat_tracker = StatTracker(log_dir=log_dir)

# get dataloaders for training and testing
# train_loader, test_loader, num_classes = \
#     build_dataset(dataset=dataset,
#                   batch_size=args.batch_size,
#                   input_dir=args.input_dir,
#                   labeled_only=True)

num_classes = 10
torch_device = torch.device('cuda')
# create new model with random parameters
model = Model(ndf=args.ndf, n_classes=num_classes, n_rkhs=args.n_rkhs,
              tclip=args.tclip, n_depth=args.n_depth, enc_size=enc_size,
              use_bn=(args.use_bn == 1))
model.init_weights(init_scale=1.0)
# restore model parameters from a checkpoint if requested
checkpoint = Checkpoint(model, args.cpt_load_path, args.output_dir, args.cpt_name)
model = model.to(torch_device)

# select which type of training to do
task = train_classifiers if args.classifiers else train_self_supervised


ckpt=torch.load('/root/amdim-public/runs_stl64_norm_BN/cifar_amdim_cpt.pth')
params = ckpt['model']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in params.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
# print(new_state_dict)
model.load_state_dict(new_state_dict)
# model.load_state_dict(params)


#%%

from datasets_bn import Dataset, build_dataset_whole, get_dataset, get_encoder_size
dataset = get_dataset(args.dataset)
train_loader, test_loader, num_classes =     build_dataset_whole(dataset=dataset,
                  batch_size=args.batch_size,
                  input_dir=args.input_dir,
                  labeled_only=True)
x_train, y_train = iter(train_loader).next()
# x_train,_=x_train
x_test, y_test = iter(test_loader).next()


#%%

# import numpy as np

# x_train=torch.from_numpy(np.load('slt_x_train.npy'))
# y_train=torch.from_numpy(np.load('slt_y_train.npy'))
# x_test=torch.from_numpy(np.load('slt_x_test.npy'))
# y_test=torch.from_numpy(np.load('slt_y_test.npy'))

import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import foolbox
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.dataset_utils import *
from lib.cifar_resnet import *
from lib.adv_model import *
from lib.dknn_attack import DKNNAttack

from lib.dknn_attack import DKNNAttack

from lib.cwl2_attack import CWL2Attack
from lib.dknn_stl3 import  DKNNL2
from lib.utils import *
from lib.lip_model import *
from lib.knn import *
from lib.nin import *
from lib.cifar10_model import *

from lib.cifar10_dcgan import Discriminator, Generator

from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion


def set_model(n_data):
    contrast = NCEAverage(128, n_data, 4096, 0.1, 0.5)
    criterion_l = NCECriterion(n_data)
    criterion_ab = NCECriterion(n_data)

    if torch.cuda.is_available():
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return contrast, criterion_ab, criterion_l


def set_optimizer(model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=0.4)
    return optimizer



layers = ['encode']
num = 10000
dknn = DKNNL2(model, x_train, y_train, x_test, y_test, layers,
              k=75, num_classes=10)

with torch.no_grad():
    y_pred = dknn.classify(x_test)
    ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
    print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))

# from lib.dknn_attack_l2_pgd import DKNNPGDAttack
# from lib.dknn_attack_l2_pgd_withinput import DKNNPGDAttack
from lib.pgd_norm_stl import DKNNPGDAttack

import pickle

attack = DKNNPGDAttack()

layer = 'encoder'


def attack_batch(x, y, batch_size, layer):
    x_a = torch.zeros_like(x)
    total_num = x.size(0)
    num_batches = total_num // batch_size
    for i in range(num_batches):
        begin = i * batch_size
        end = (i + 1) * batch_size
        #         x_a[begin:end] = attack(
        #             dknn, x[begin:end], y[begin:end],
        #             guide_layer=layer, m=300, binary_search_steps=10,
        #             max_iterations=500, learning_rate=1e-2, initial_const=1e-3,
        #             abort_early=False, random_start=True, guide=2)
        x_a[begin:end] = attack(
            dknn, x[begin:end], y[begin:end],
            guide_layer=layer, m=300, binary_search_steps=20,
            max_iterations=1000, learning_rate=1e-2, initial_const=1e-3,
            abort_early=False, random_start=True, guide=2,epsilon=0.06, step = 20, step_size = 0.005)
    return x_a


num = 100
ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
x_adv = attack_batch(x_test[ind][:num].cuda(), y_test[ind][:num], 10,layer)
model_name = 'amdim003_stl'
pickle.dump(x_adv.cpu().detach(), open('x_adv_' + model_name + '.p', 'wb'))
y_pred = dknn.classify(x_adv)
acc = (y_pred.argmax(1) == y_test[ind][:num].numpy()).sum() / len(y_pred)
print(acc)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
l_loss_meter = AverageMeter()
ab_loss_meter = AverageMeter()
l_prob_meter = AverageMeter()
ab_prob_meter = AverageMeter()


# optimizer = set_optimizer(model)
mods_inf = [m for m in model.info_modules]
optimizer = optim.Adam(
    [{'params': mod.parameters(), 'lr': 0.0001} for mod in mods_inf],
    betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)



batch_size =100

for i in range(400):
    print(i)
    begin = i * batch_size
    end = (i + 1) * batch_size
    x_ori = x_train[begin:end].cuda()
    x_a = torch.zeros_like(x_ori).cuda()
    y_ori = y_train[begin:end].cuda()
#         model.eval()
    #         with torch.no_grad():

    # num = 20000
    dknn = DKNNL2(model, x_train, y_train, x_test, y_test, layers, k=75, num_classes=10)

    x_a = attack(
        dknn, x_ori, y_ori,
        guide_layer=layer, m=300, binary_search_steps=10,
        max_iterations=10, learning_rate=1e-2, initial_const=1e-5,
        abort_early=False, random_start=True, guide=2,epsilon=0.03, step = 10, step_size = 0.005)

    y_pred = dknn.classify(x_a)
    ind = np.where(y_pred.argmax(1) != y_ori.cpu().numpy())[0]
    index = torch.LongTensor(list(range(begin, end)))
    x_a = x_a[ind]
    x_ori = x_ori[ind]
    y_ori = y_ori[ind]
    index = index[ind]

    model.train()

    bsz = x_ori.size(0)
    inputs = x_ori.float()
    inputs_adv = x_a.detach().float()
    if torch.cuda.is_available():
        index = index.cuda()
        inputs = inputs.cuda()
        inputs_adv = inputs_adv.cuda()

        # ===================forward=====================
    if inputs.shape[0] == 1:
        inputs = torch.cat((inputs, inputs), 0)
        inputs_adv = torch.cat((inputs_adv, inputs_adv), 0)
        res_dict = model(x1=inputs, x2=inputs_adv, class_only=False)
        lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        # compute costs for all self-supervised tasks
        loss_g2l = (res_dict['g2l_1t5'] +
                    res_dict['g2l_1t7'] +
                    res_dict['g2l_5t5'])
        loss_inf = loss_g2l #+ res_dict['lgt_reg']
    else:
        res_dict = model(x1=inputs, x2=inputs_adv, class_only=False)
        lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        # compute costs for all self-supervised tasks
        loss_g2l = (res_dict['g2l_1t5'] +
                    res_dict['g2l_1t7'] +
                    res_dict['g2l_5t5'])
        loss_inf = loss_g2l #+ res_dict['lgt_reg']

    loss = loss_inf

    # ===================backward=====================
    optimizer.zero_grad()
    mixed_precision.backward(loss, optimizer)
    optimizer.step()

    # ===================meters=====================
    #         losses.update(loss.item(), bsz)

    #     batch_time.update(time.time() - end)
    #     end = time.time()



    # print info
    if (i + 1) % 1 == 0:
        print('loss: {}'.format(loss.item()))


#         if i == 20:
#             print('==> Saving...')
#             state = {
#                 'model': model.state_dict(),
#                 'epoch': epoch,
#             }
#             torch.save(state, 'save_models/new_amdim128_pgd_mul_ckpt_20_amd_{epoch}.pth'.format(epoch=epoch))
#         with torch.no_grad():
#             y_pred = dknn.classify(x_test)
#             ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
#             print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))
#             y_pred = dknn.classify(inputs_adv)
#             acc = (y_pred.argmax(1) == y_ori[ind].cpu().numpy()).sum() / len(y_pred)
#             print(acc)



#         if i == 50:
    if i%5==0:
        print('==> Saving...')
        state = {
            'model': model.state_dict(),
            'i': i,
        }
        torch.save(state, 'stl/lr0001_003_SAT_{}.pth'.format(i))



    dknn = DKNNL2(model, x_train, y_train, x_test, y_test, layers, k=75, num_classes=10)
    with torch.no_grad():
        y_pred = dknn.classify(x_test)
        ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
        print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))

        y_pred = dknn.classify(x_a)
        print('adversarial example')
        print((y_pred.argmax(1) == y_ori.cpu().numpy()).sum()/x_a.shape[0])

            
#     print('==> Saving...')
#     state = {
#         'model': model.state_dict(),
#         'epoch': epoch,
#     }
#     torch.save(state, 'save_models/new_amdim128_pgd_mul_ckpt_50_amd_{epoch}.pth'.format(epoch=epoch))

