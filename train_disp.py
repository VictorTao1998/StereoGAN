import time
import os
import argparse
import sys
import itertools
import numpy as np
from scipy import misc

import torch
import torch.nn as nn 
import torch.optim as  optim
import torch.nn.functional as F
from models.loss import warp_loss, model_loss0
from models.dispnet import dispnetcorr
from models.gan_nets import GeneratorResNet, Discriminator, weights_init_normal
from datasets.messytable import MessytableDataset
from datasets.messytable_test import MessytableTestDataset_TEST

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.util import AverageMeter
from utils.util import load_multi_gpu_checkpoint, load_checkpoint
from utils.metric_utils.metrics import *
from utils import pytorch_ssim
from utils.config import cfg
from utils.data_util import *

def val(valloader, net, writer, epoch=1, board_save=True):
    print("begin validation")
    net.eval()
    EPEs, D1s, Thres1s, Thres2s, Thres3s = 0, 0, 0, 0, 0
    i = 0
    for sample in valloader:
        left_img = sample['img_L'].cuda()
        right_img = sample['img_R'].cuda()
        disp_gt = sample['img_disp_l'].cuda()
        left_img = F.interpolate(left_img, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)
        right_img = F.interpolate(right_img, scale_factor=0.5, mode='bilinear',
                                recompute_scale_factor=False, align_corners=False)
        disp_gt = F.interpolate(disp_gt, scale_factor=0.5, mode='nearest',
                            recompute_scale_factor=False)  # [bs, 1, H, W]
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        disp_est = net(left_img, right_img)[0].squeeze(1)
        #print(disp_est.shape, disp_gt.shape, mask.shape)
        EPEs += EPE_metric(disp_est, disp_gt[0], mask[0])
        D1s += D1_metric(disp_est, disp_gt[0], mask[0])
        Thres1s += Thres_metric(disp_est, disp_gt[0], mask[0], 2.0)
        Thres2s += Thres_metric(disp_est, disp_gt[0], mask[0], 4.0)
        Thres3s += Thres_metric(disp_est, disp_gt[0], mask[0], 5.0)
    if board_save:
        writer.add_scalar("val/EPE", EPEs/i, epoch)
        writer.add_scalar("val/D1", D1s/i, epoch)
        writer.add_scalar("val/Thres2", Thres1s/i, epoch)
        writer.add_scalar("val/Thres4", Thres2s/i, epoch)
        writer.add_scalar("val/Thres5", Thres3s/i, epoch)
    return EPEs/i, D1s/i

def train(args,cfg):
    writer = SummaryWriter(args.writer)
    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    argsDict = args.__dict__
    for k,v in argsDict.items():
        writer.add_text('hyperparameter', '{} : {}'.format(str(k), str(v)))

    print_freq = args.print_freq
    test_freq = 1
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    
    input_shape = (1, cfg.ARGS.CROP_HEIGHT, cfg.ARGS.CROP_WIDTH)
    net = dispnetcorr(args.maxdisp)


    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_dispnet_path:
                net = load_multi_gpu_checkpoint(net, args.load_dispnet_path, 'model')
            else:
                net.apply(weights_init_normal)
        else:
            if args.load_dispnet_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
    else:
        net.apply(weights_init_normal)
    # optimizer = optim.SGD(params, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr_rate, betas=(0.9, 0.999))

    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))


    net.to(device)


    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_identity = torch.nn.L1Loss().cuda()
    ssim_loss = pytorch_ssim.SSIM()

    # data loader
    train_dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=False, color_jitter=False, debug=False, sub=600)
    val_dataset = MessytableTestDataset_TEST(cfg.REAL.TRAIN, debug=False, sub=100, onReal=True)

    TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)

    ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)
    print(cfg.SOLVER.BATCH_SIZE,cfg.SOLVER.NUM_WORKER)
    
    #if args.source_dataset == 'driving':
    #    dataset = ImageDataset(height=args.img_height, width=args.img_width)
    #elif args.source_dataset == 'synthia':
    #    dataset = ImageDataset2(height=args.img_height, width=args.img_width)
    #else:
    #    raise "No suportive dataset"
    
    #trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    #valdataset = ValJointImageDataset()
    #valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    ## debug only
    #with torch.no_grad():
    #    l1_test_loss, out_val = val(valloader, net, G_AB, None, writer, epoch=0, board_save=True)
    #    val_loss_meter.update(l1_test_loss)
    #    print('Val epoch[{}/{}] loss: {}'.format(0, args.total_epochs, l1_test_loss))

    print('begin training...')
    best_val_d1 = 1.
    best_val_epe = 100.
    for epoch in range(cfg.SOLVER.EPOCHS):
        #net.train()
        #G_AB.train()

        n_iter = 0
        running_loss = 0.
        t = time.time()
        # custom lr decay, or warm-up
        lr = args.lr_rate
        if epoch >= int(args.lrepochs.split(':')[0]):
            lr = lr / int(args.lrepochs.split(':')[1])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, batch in enumerate(TrainImgLoader):
            n_iter += 1
            leftA = batch['img_sim_L'].to(device)
            rightA = batch['img_sim_R'].to(device)
            leftB = batch['img_real_L'].to(device)
            rightB = batch['img_real_R'].to(device)
            dispA = batch['img_disp_l'].to(device)
            dispB = batch['img_disp_r'].to(device) 
            leftB = F.interpolate(leftB, scale_factor=0.5, mode='bilinear',
                             recompute_scale_factor=False, align_corners=False)
            rightB = F.interpolate(rightB, scale_factor=0.5, mode='bilinear',
                                    recompute_scale_factor=False, align_corners=False)
            dispA = F.interpolate(dispA, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
            dispB = F.interpolate(dispB, scale_factor=0.5, mode='nearest',
                                    recompute_scale_factor=False)  # [bs, 1, H, W]
            #if args.warp_op:
            #    img_disp_r = sample['img_disp_r'].to(cuda_device)
            #    img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
            #                            recompute_scale_factor=False)
            #    disp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
            #    del img_disp_r
            #dispA = dispA.unsqueeze(1).float()

            # train disp net
            net.train()

            optimizer.zero_grad()
            disp_ests = net(leftA, rightA)
            mask = (dispA < args.maxdisp) & (dispA > 0)
            #print(mask.dtype)
            loss0 = model_loss0(disp_ests, dispA, mask)
            #print(mask.dtype)


            #print(mask.dtype)

            loss = loss0
            loss.backward()
            optimizer.step()

            if i % print_freq == print_freq - 1:
                print('epoch[{}/{}]  step[{}/{}]  loss: {}'.format(epoch, cfg.SOLVER.EPOCHS, i, len(TrainImgLoader), loss.item() ))
                train_loss_meter.update(running_loss / print_freq)
                #writer.add_scalar('loss/trainloss avg_meter', train_loss_meter.val, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp', loss0, train_loss_meter.count * print_freq)
                save_images(writer, 'train', {'img_L':[leftA.detach().cpu()]}, i)
                save_images(writer, 'train', {'img_R':[rightA.detach().cpu()]}, i)
                save_images(writer, 'train', {'disp_gt':[dispA[0].detach().cpu()]}, i)
                save_images(writer, 'train', {'disp_pred':[disp[0].detach().cpu() for disp in disp_ests]}, i)
                #writer.add_image('pred/gt_disp', dispA[0].detach().cpu(), i)
                #writer.add_image('pred/pred_disp', disp_ests[0][0].detach().cpu(), i)

            #print(mask.dtype)

        with torch.no_grad():
            EPE, D1 = val(ValImgLoader, net, writer, epoch=epoch, board_save=True)

        t1 = time.time()
        print('epoch:{}, D1:{:.6f}, EPE:{:.6f}, cost time:{} '.format(epoch, D1, EPE, t1-t))

        if (epoch % args.save_interval == 0) or D1 < best_val_d1 or EPE < best_val_epe:
            best_val_d1 = D1
            best_val_epe = EPE
            torch.save({
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.checkpoint_save_path + '/ep' + str(epoch) + '_D1_{:.4f}_EPE{:.4f}'.format(D1, EPE) + '.pth.rar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # training
    parser.add_argument('--lr_rate', nargs='?', type=float, default=1e-3, help='learning rate for dispnetc')
    parser.add_argument('--lrepochs', type=str, default='30:1', help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--lr_gan', nargs='?', type=float, default=2e-4, help='learning rate for GAN')
    parser.add_argument('--train_ratio_gan', nargs='?', type=int, default=5, help='training ratio disp:gan=5:1')
    parser.add_argument('--save_interval', nargs='?', type=int, default='10')
    parser.add_argument('--model_type', nargs='?', type=str, default='dispnetc')
    parser.add_argument('--maxdisp', type=int, default=192)

    # hyper params
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--alpha_ssim', type=float, default=0.85)
    parser.add_argument('--lambda_id', type=float, default=5)
    parser.add_argument('--lambda_ms', type=float, default=1)
    parser.add_argument('--lambda_warp', type=float, default=0)
    parser.add_argument('--lambda_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_disp_warp', type=float, default=0)
    parser.add_argument('--lambda_disp_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_corr', type=float, default=10)

    # load & save checkpoints
    parser.add_argument('--load_checkpoints', nargs='?', type=int, default=0, help='load from ckp(saved by Pytorch)')
    parser.add_argument('--load_from_mgpus_model', nargs='?', type=int, default=0, help='load ckp which is saved by mgus(nn.DataParallel)')
    parser.add_argument('--load_gan_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--load_dispnet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--checkpoint_save_path', nargs='?', type=str, default='checkpoints/best_checkpoint.pth.tar')

    # tensorboard, print freq
    parser.add_argument('--writer', nargs='?', type=str, default='StereoGAN')
    parser.add_argument('--print_freq', '-p', default=150, type=int, metavar='N', help='print frequency (default: 150)')

    # other
    parser.add_argument('--use_multi_gpu', nargs='?', type=int, default=0, help='the number of multi gpu to use')
    parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                    metavar='FILE', help='Config files')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    #set_random_seed(args.seed)
    train(args,cfg)
