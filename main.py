import argparse 
import torch.nn as nn
import torchvision.transforms as transforms
from model import Transformer
import numpy as np
import os
import torch
from sklearn.model_selection import KFold
import torch.utils.data as data_utils
from utils import create_optimizer
from eval import train_model, test_model
from DataLoader import csvloader

current_dir = os.getcwd()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Attention model')

parser.add_argument('--dataroot', type=str,
                    help='path to dataset')

parser.add_argument('--log-dir', default = os.getcwd(),
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default=os.getcwd()+'with_cross_entropy_run-optim_adagrad-lr{}-wd{}-N{}-heads{}/checkpoint_1.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Training options
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=16, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.125 , metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4)')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--CV', default=5, type=int,
                    metavar='cv', help='Number of folds')
parser.add_argument('--N', default=2, type=int,
                    metavar='cv', help='N is the variable for the number of layers there will be')
parser.add_argument('--heads', default=2, type=int,
                    metavar='heads', help='Multi-heads attention')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

np.random.seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.log_dir):
	os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True

LOG_DIR = args.log_dir + '/with_cross_entropy_run-optim_{}-lr{}-wd{}-N{}-heads{}'\
    .format(args.optimizer, args.lr, args.wd, args.N, args.heads)


d_model = 20
transform = transforms.Compose([torch.Tensor()])
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
kf = KFold(n_splits=args.CV)
def main():
    months = os.listdir(args.dataroot)
    for month_i in months:
        bugs = os.listdir(args.dataroot + '/' + month_i + '/Nonover')
        for bug_i in bugs:
            if bug_i == 'Bugs 1':
                dirs = args.dataroot + '/' + month_i + '/Nonover/' + '/' + bug_i
                print("Month: {} Bug no.: {}".format(month_i, bug_i))
                files = os.listdir(dirs)
                i = 0
                for train_index, test_index in kf.split(files):
                    i = i + 1
                    train_dir = csvloader(root_dir = dirs, idx = train_index)
                    test_dir = csvloader(root_dir = dirs, idx = test_index, mode = 'Test')
                    DigsiGitData = csvloader(root_dir = Gitpath, idx = idxs, mode = 'Test')
                    train_loader = torch.utils.data.DataLoader(train_dir, 
                        batch_size=1, shuffle=True, **kwargs)
                    test_loader = torch.utils.data.DataLoader(test_dir, 
                        batch_size=1, shuffle=False, **kwargs)
                    model = Transformer(7, d_model, args.N, args.heads)
                    for p in model.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                    if args.cuda:
                        model.cuda()
                    optimizer = create_optimizer(model, args.lr, args)
                    start = args.start_epoch
                    end = start + args.epochs
                    print('starting') 
                    
                    if os.path.isfile(args.resume):
                        print("LOADING MODEL")
                        checkpoint = torch.load(args.resume)
                        args.start_epoch = checkpoint['epoch']
                        model.load_state_dict(checkpoint['state_dict'])
                    
                    for epoch in range(start, end):
                        train_model(train_loader, model, optimizer, epoch, i, LOG_DIR, args)
                        if epoch%2 == 0:
                            test_model(test_loader, model, epoch, LOG_DIR, args)
                        

if __name__ == '__main__':
    main()




