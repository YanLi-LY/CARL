import sys
import time
import warnings
import numpy as np
import copy
import torch.nn.parallel
from collections import OrderedDict

warnings.filterwarnings('ignore')

import torch.nn as nn
from torch import optim
from torchvision.models import vgg16
from torch.autograd import Variable
from torch.backends import cudnn

from option import *
from utils import *
from data_utils import *
from metrics import *
from models.model import backbone
from losses.ContrastLoss import LossNetwork as ContrastLoss

create_dir('./Log/')
trainLogger = open('./Log/train.log', 'a+')
trainLogger.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
trainLogger.write('\n')

start_time=time.time()

def train(backbone_model, ema_model, train_loader, test_loader, optim, criterion):
    start_epoch = 0
    max_ssim_1, max_ssim_2 = 0, 0
    max_psnr_1, max_psnr_2 = 0, 0
    total_iters = len(train_loader)
    losses = []
    set_consistloss = True

    if opt.if_load_pre and os.path.exists(opt.pre_model_path):
        print(f'load_pretrain_model from {opt.pre_model_path}')
        ckp = torch.load(opt.pre_model_path)
        backbone_model.load_state_dict(ckp['backbone_model'])
        ema_model.load_state_dict(ckp['ema_model'])
        max_ssim_1 = ckp['max_ssim_1']
        max_psnr_1 = ckp['max_psnr_1']
        max_ssim_2 = ckp['max_ssim_2']
        max_psnr_2= ckp['max_psnr_2']
        print(f'start_epoch:{start_epoch} start training ---')
        print(f'max_ssim_1: {max_ssim_1}, max_psnr_1: {max_psnr_1}, max_ssim_2: {max_ssim_2}, max_psnr_2: {max_psnr_2}')
    else :
        print('train from scratch *** ')

    for param in ema_model.parameters():
        param.requires_grad = False
    for param in backbone_model.parameters():
        param.requires_grad = True

    for epoch in range(start_epoch+1, opt.epochs+1):
        backbone_model.train()

        lr = lr_schedule_cosdecay(epoch, opt.epochs, opt.lr) if opt.lr_sche else opt.lr
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        for batch_id, data in enumerate(train_loader):
            # if batch_id == 5:
            #     break
            optim.zero_grad()
            haze_1, haze_2, clear_img, other_method_out, haze_1_name, haze_2_name = data
            clear_img = clear_img.to(device)
            haze_1 = Variable(haze_1).to(device)
            haze_2 = Variable(haze_2).to(device)
            other_method_out = Variable(other_method_out).to(device)
            if not opt.ema:
                out = backbone_model(haze_1)
                l1_loss = criterion[0](out, clear_img)
                if opt.perloss:
                    per_loss = criterion[1](out, clear_img)
                    loss = l1_loss + 0.04 * per_loss
                if opt.contrastloss:
                    contrast_loss = criterion[1](out, clear_img, haze_1)
                    loss = l1_loss + contrast_loss
            else:
                out_1 = backbone_model(haze_1)
                with torch.no_grad():
                    out_2 = ema_model(haze_2)
                l1_loss = criterion[0](out_1, clear_img)

                if set_consistloss:
                    consist_loss = criterion[0](out_1, out_2.detach_())
                    loss = l1_loss + consist_loss
                else:
                    loss = l1_loss

                if opt.perloss:
                    per_loss = criterion[1](out_1, clear_img)
                    loss = loss + 0.04 * per_loss
                if opt.contrastloss:
                    contrast_loss = criterion[1](out_1, clear_img, haze_1, haze_2, other_method_out)
                    loss = loss + contrast_loss

            loss.backward()
            optim.step()
            losses.append(loss.item())

            if opt.ema:
                state_dict_backbone = backbone_model.state_dict()
                state_dict_ema_model = ema_model.state_dict()
                for (k_backbone, v_backbone), (k_ema, v_ema) in zip(state_dict_backbone.items(), state_dict_ema_model.items()):
                    assert k_backbone == k_ema
                    assert v_backbone.shape == v_ema.shape
                    if 'num_batches_tracked' in k_ema:
                        v_ema.copy_(v_backbone)
                    else:
                        v_ema.copy_(v_ema * opt.momentum + (1. - opt.momentum) * v_backbone) # momentum=0.999

            if set_consistloss:
                print(f'\rtrain loss : {loss.item():.5f} | l1_loss : {l1_loss.item():.5f} | contrast_loss : {contrast_loss.item():.5f} | consist_loss : {consist_loss.item():.5f} '
                  f'| epoch : {epoch}/{opt.epochs} | iter : {batch_id + 1}/{total_iters} | lr : {lr :.5f} | time_used : {(time.time()-start_time)/60 :.1f}',end='',flush=True)
            else:
                print(
                    f'\rtrain loss : {loss.item():.5f} | l1_loss : {l1_loss.item():.5f} | contrast_loss : {contrast_loss.item():.5f}'
                    f'| epoch : {epoch}/{opt.epochs} | iter : {batch_id + 1}/{total_iters} | lr : {lr :.5f} | time_used : {(time.time() - start_time) / 60 :.1f}',
                    end='', flush=True)

            if epoch % opt.eval_epoch == 0 and (batch_id + 1) % (total_iters) == 0:
                with torch.no_grad():
                    ssim_1_eval, psnr_1_eval, ssim_2_eval, psnr_2_eval = test(backbone_model, ema_model, test_loader)

                if set_consistloss:
                    print(f'\nepoch : {epoch}  | train loss : {loss.item():.5f} | l1_loss : {l1_loss.item():.5f} | contrast_loss : {contrast_loss.item():.5f} | consist_loss : {consist_loss.item():.5f} | lr : {lr :.5f} '
                        f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')
                    trainLogger.write(f'\nepoch : {epoch}  | train loss : {loss.item():.5f} | l1_loss : {l1_loss.item():.5f} | contrast_loss : {contrast_loss.item():.5f} | consist_loss : {consist_loss.item():.5f} | lr : {lr :.5f} '
                        f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')
                else:
                    print(f'\nepoch : {epoch}  | train loss : {loss.item():.5f} | lr : {lr :.5f} '
                        f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')
                    trainLogger.write(f'\nepoch : {epoch}  | train loss : {loss.item():.5f} | lr : {lr :.5f} '
                        f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')

                if ssim_1_eval > max_ssim_1 and psnr_1_eval > max_psnr_1:
                    max_ssim_1 = max(max_ssim_1, ssim_1_eval)
                    max_psnr_1 = max(max_psnr_1, psnr_1_eval)
                if ssim_2_eval >= max_ssim_2 and psnr_2_eval >= max_psnr_2 :
                    max_ssim_2 = max(max_ssim_2, ssim_2_eval)
                    max_psnr_2 = max(max_psnr_2, psnr_2_eval)
                save_model_path = os.path.join(opt.save_model_dir, str(opt.trainset) + '_epoch_' + str(epoch) +  '.pk')
                torch.save({
                    'epoch': epoch,
                    'losses': losses,
                    'max_psnr_1': max_psnr_1,
                    'max_ssim_1': max_ssim_1,
                    'max_psnr_2': max_psnr_2,
                    'max_ssim_2': max_ssim_2,
                    'backbone_model': backbone_model.state_dict(),
                    'ema_model': ema_model.state_dict()
                }, save_model_path)
                print(f'\nmodel saved at epoch : {epoch} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_2 : {max_psnr_2:.4f} | max_ssim_2 : {max_ssim_2:.4f}')
                trainLogger.write(f'\nmodel saved at epoch : {epoch} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_2 : {max_psnr_2:.4f} | max_ssim_2 : {max_ssim_2:.4f}')


def test(backbone_model, ema_model, test_loader):
    backbone_model.eval()
    ema_model.eval()
    torch.cuda.empty_cache()
    ssims_1, ssims_2 = [], []
    psnrs_1, psnrs_2 = [], []
    for val_batch_id, val_data in enumerate(test_loader):
        # if val_batch_id == 5:
        #     break
        input, target = val_data
        input = input.to(device)
        target = target.to(device)
        if opt.ema:
            pred_1 = backbone_model(input)
            pred_2 = ema_model(input)
            ssim_1 = ssim(pred_1, target).item()
            psnr_1 = psnr(pred_1, target)
            ssim_2 = ssim(pred_2, target).item()
            psnr_2 = psnr(pred_2, target)
            ssims_1.append(ssim_1)
            psnrs_1.append(psnr_1)
            ssims_2.append(ssim_2)
            psnrs_2.append(psnr_2)

        else:
            pred_1 = backbone_model(input)
            ssim_1 = ssim(pred_1, target).item()
            psnr_1 = psnr(pred_1, target)
            ssims_1.append(ssim_1)
            psnrs_1.append(psnr_1)
    return np.mean(ssims_1), np.mean(psnrs_1), np.mean(ssims_2), np.mean(psnrs_2)


if __name__ == "__main__":
    # dataloader
    train_loader = train_loader
    test_loader = test_loader

    # create model
    backbone_model = backbone(gps=opt.gps, blocks=opt.blocks)
    ema_model =  backbone(gps=opt.gps, blocks=opt.blocks)
    device = torch.device(opt.device)
    backbone_model = backbone_model.to(device)
    ema_model = ema_model.to(device)
    backbone_model = torch.nn.DataParallel(backbone_model, device_ids=[0, 1])
    ema_model = torch.nn.DataParallel(ema_model, device_ids=[0, 1])

    # define loss function
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    if opt.contrastloss:
        criterion.append(ContrastLoss().to(device))

    # define optimizer
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, backbone_model.parameters()), lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)

    # start training
    train(backbone_model, ema_model, train_loader, test_loader, optimizer, criterion)

    # Ending
    trainLogger.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


