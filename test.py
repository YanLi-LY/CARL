import time

import torch
from torchvision import utils as vutils

from metrics import *
from option import *
from utils import *
from data_utils import *

from models.model import backbone

device = torch.device(opt.device)
create_dir('./testLog/')
trainLogger = open('./testLog/test.log', 'a+')
trainLogger.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
trainLogger.write('\n')

test_result_dir = './testResult/'
create_dir(test_result_dir)

def test(backbone_model, ema_model, test_loader):
    backbone_model.eval()
    ema_model.eval()
    torch.cuda.empty_cache()
    ssims_1, ssims_2 = [], []
    psnrs_1, psnrs_2 = [], []
    for val_batch_id, val_data in enumerate(test_loader):
        input, target, name = val_data

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

            vutils.save_image(torch.cat((pred_1, pred_2, target), 0), os.path.join(test_result_dir, f'{name}.png'))

        else:
            pred_1 = backbone_model(input)
            ssim_1 = ssim(pred_1, target).item()
            psnr_1 = psnr(pred_1, target)
            ssims_1.append(ssim_1)
            psnrs_1.append(psnr_1)
    return np.mean(ssims_1), np.mean(psnrs_1), np.mean(ssims_2), np.mean(psnrs_2)


if __name__ == "__main__":
    test_loader = test_loader
    backbone_model = backbone(gps=opt.gps, blocks=opt.blocks)
    ema_model =  backbone(gps=opt.gps, blocks=opt.blocks)
    device = torch.device(opt.device)
    backbone_model = backbone_model.to(device)
    ema_model = ema_model.to(device)

    pre_model_path = ''
    ckp = torch.load(pre_model_path)
    backbone_model.load_state_dict(ckp['backbone_model'])
    ema_model.load_state_dict(ckp['ema_model'])

    for param in ema_model.parameters():
        param.requires_grad = False
    for param in backbone_model.parameters():
        param.requires_grad = False

    backbone_model.eval()
    ema_model.eval()
    with torch.no_grad():
        ssim_1_eval, psnr_1_eval, ssim_2_eval, psnr_2_eval = test(backbone_model, ema_model, test_loader)

    print(f'\nssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')
    trainLogger.write(f'\nssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')

    trainLogger.write(time.strftime('\n%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    trainLogger.close()
