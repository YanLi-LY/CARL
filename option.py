import os
import warnings
import argparse
from utils import *

warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--if_load_pre', type=bool, default=True)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--pre_model_path', type=str, default='', help='the pre_train model .pk')
parser.add_argument('--save_model_dir', type=str, default='./checkpoints/', help='save train model dir.pk')
parser.add_argument('--trainset', type=str, default='ITS')
parser.add_argument('--testset', type=str, default='SOTS-Indoor')
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=20, help='residual_blocks')
parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--crop', type=str, default=True)
parser.add_argument('--crop_size', type=int, default=240, help='Takes effect when using --crop ')
parser.add_argument('--lr_sche', type=str, default=True, help='lr cos schedule')
parser.add_argument('--contrastloss', type=str, default=True, help='Contrastive Loss')
parser.add_argument('--ema', type=str, default=True, help='use ema')
parser.add_argument('--momentum', type=float, default=0.999, help='ema decay rate')

opt = parser.parse_args()

print(opt)

create_dir(opt.save_model_dir)