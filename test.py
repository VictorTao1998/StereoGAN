import time
import os
import argparse
import sys
import itertools
import numpy as np
from scipy import misc

from tqdm import tqdm

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
from utils.val_util import get_time_string, setup_logger, depth_error_img, disp_error_img, save_images_grid, save_images
from utils.test_util import load_from_dataparallel_model, save_img, save_gan_img, save_obj_err_file
from utils.warp_ops import apply_disparity_cu
import torchvision.transforms as transforms


from utils.cascade_metrics import compute_err_metric, compute_obj_err

real_obj_id = [4, 5, 7, 9, 13, 14, 15, 16]

def test_sample(net, val_loader, logger, log_dir, summary_writer, args, cfg):
    net.eval()


    total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                         'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}
    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, 'pred_disp'))
    os.mkdir(os.path.join(log_dir, 'gt_disp'))
    os.mkdir(os.path.join(log_dir, 'pred_disp_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'pred_depth'))
    os.mkdir(os.path.join(log_dir, 'gt_depth'))
    os.mkdir(os.path.join(log_dir, 'pred_depth_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'gan'))
    #os.mkdir(args.output + "/feature")

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data['img_L'].cuda()    # [bs, 1, H, W]
        img_R = data['img_R'].cuda()

        img_disp_l = data['img_disp_l'].cuda()
        img_depth_l = data['img_depth_l'].cuda()
        img_depth_realsense = data['img_depth_realsense'].cuda()
        img_label = data['img_label'].cuda()
        img_focal_length = data['focal_length'].cuda()
        img_baseline = data['baseline'].cuda()
        prefix = data['prefix'][0]

        img_disp_l = F.interpolate(img_disp_l, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_depth_l = F.interpolate(img_depth_l, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_depth_realsense = F.interpolate(img_depth_realsense, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_label = F.interpolate(img_label, (540, 960), mode='nearest',
                             recompute_scale_factor=False).type(torch.int)

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data['img_disp_r'].cuda()
            img_depth_r = data['img_depth_r'].cuda()
            img_disp_r = F.interpolate(img_disp_r, (540, 960), mode='nearest',
                                       recompute_scale_factor=False)
            img_depth_r = F.interpolate(img_depth_r, (540, 960), mode='nearest',
                                        recompute_scale_factor=False)
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(img_depth_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)

            

        #img_L_real = data['img_L_real'].cuda()    # [bs, 1, H, W]
        #img_L_real = F.interpolate(img_L_real, (540, 960), mode='bilinear',
        #                      recompute_scale_factor=False, align_corners=False)
        img_L_o = F.interpolate(img_L, (540, 960), mode='bilinear',
                            recompute_scale_factor=False, align_corners=False)
        img_R_o = F.interpolate(img_R, (540, 960), mode='bilinear',
                            recompute_scale_factor=False, align_corners=False)
        
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540
        img_L = F.pad(img_L_o, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        img_R = F.pad(img_R_o, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        

        # Save gan results
        #print(img_L_o.shape, img_R_o.shape, gan_model.real_A_L[:,0,:,:][:,None,:,:].shape, prefix)

        img_inputs = {
            'input': {
                'input_L': img_L_o, 'input_R': img_R_o
            }
        }
        #save_images_grid(summary_writer, 'test_gan', input_sample, iteration)
        save_images_grid(summary_writer, 'test_gan', img_inputs, iteration)

        # Pad the imput image and depth disp image to 960 * 544
        #right_pad = cfg.REAL.PAD_WIDTH - 960
        #top_pad = cfg.REAL.PAD_HEIGHT - 540
        #img_L = F.pad(img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        #img_R = F.pad(img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)

        if args.exclude_bg:
            print("bg flag is: ", args.exclude_bg)
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * img_ground_mask
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0)

        # Exclude uncertain pixel from realsense_depth_pred
        realsense_zeros_mask = img_depth_realsense > 0
        if args.exclude_zeros:
            print("zero flag is: ", args.exclude_zeros)
            mask = mask * realsense_zeros_mask
        mask = mask.type(torch.bool)

        ground_mask = torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()

        with torch.no_grad():
            pred_disp = net(img_L, img_R)[0]

        pred_disp = pred_disp[:, :, top_pad:, :]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m

        # Get loss metric
        err_metrics = compute_err_metric(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                         img_baseline, mask)
        for k in total_err_metrics.keys():
            total_err_metrics[k] += err_metrics[k]
        logger.info(f'Test instance {prefix} - {err_metrics}')

        # Get object error
        obj_disp_err, obj_depth_err, obj_depth_4_err, obj_count = compute_obj_err(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                                     img_baseline, img_label, mask, cfg.SPLIT.OBJ_NUM)
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_depth_4_err += obj_depth_4_err
        total_obj_count += obj_count

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        pred_disp_np[ground_mask] = -1
        pred_disp_np_o = torch.tensor(pred_disp_np)[None,:,:]

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_disp_np[ground_mask] = -1
        gt_disp_np_o = torch.tensor(gt_disp_np)[None,:,:]

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, mask)
        #print("disperr: ", pred_disp_err_np.shape)

        # Get depth image
        pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        pred_depth_np[ground_mask] = -1
        pred_depth_np_o = torch.tensor(pred_depth_np)[None,:,:]

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_depth_np[ground_mask] = -1
        gt_depth_np_o = torch.tensor(gt_depth_np)[None,:,:]
        #print('LOG shape: ', img_depth_l.shape, gt_depth_np.shape)

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, mask)
        #print("deperr: ", pred_depth_err_np.shape)

        # Save images
        image_test_output = {'pred_disp': pred_disp_np_o, 'gt_disp': gt_disp_np_o, 'pred_depth': pred_depth_np_o, 'gt_depth': gt_depth_np_o}
        save_images(summary_writer, 'test_psmnet', image_test_output, iteration)
        save_img(log_dir, prefix, pred_disp_np, gt_disp_np, pred_disp_err_np,
                 pred_depth_np, gt_depth_np, pred_depth_err_np)

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f'\nTest on {len(val_loader)} instances\n {total_err_metrics}')

    # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    total_obj_depth_4_err /= total_obj_count
    save_obj_err_file(total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, log_dir)

    logger.info(f'Successfully saved object error to obj_err.txt')

    # Get error on real and 3d printed objects
    real_depth_error = 0
    real_depth_error_4mm = 0
    printed_depth_error = 0
    printed_depth_error_4mm = 0
    for i in range(cfg.SPLIT.OBJ_NUM):
        if i in real_obj_id:
            real_depth_error += total_obj_depth_err[i]
            real_depth_error_4mm += total_obj_depth_4_err[i]
        else:
            printed_depth_error += total_obj_depth_err[i]
            printed_depth_error_4mm += total_obj_depth_4_err[i]
    real_depth_error /= len(real_obj_id)
    real_depth_error_4mm /= len(real_obj_id)
    printed_depth_error /= (cfg.SPLIT.OBJ_NUM - len(real_obj_id))
    printed_depth_error_4mm /= (cfg.SPLIT.OBJ_NUM - len(real_obj_id))

    logger.info(f'Real objects - absolute depth error: {real_depth_error}, depth 4mm: {real_depth_error_4mm} \n'
                f'3D printed objects - absolute depth error {printed_depth_error}, depth 4mm: {printed_depth_error_4mm}')

def test_sim(net, G_AB, G_BA, val_loader, logger, log_dir, summary_writer, args, cfg):
    net.eval()
    G_AB.eval()
    G_BA.eval()


    total_err_metrics = {'epe': 0, 'bad1': 0, 'bad2': 0,
                         'depth_abs_err': 0, 'depth_err2': 0, 'depth_err4': 0, 'depth_err8': 0}
    total_obj_disp_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_depth_4_err = np.zeros(cfg.SPLIT.OBJ_NUM)
    total_obj_count = np.zeros(cfg.SPLIT.OBJ_NUM)
    os.mkdir(os.path.join(log_dir, 'pred_disp'))
    os.mkdir(os.path.join(log_dir, 'gt_disp'))
    os.mkdir(os.path.join(log_dir, 'pred_disp_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'pred_depth'))
    os.mkdir(os.path.join(log_dir, 'gt_depth'))
    os.mkdir(os.path.join(log_dir, 'pred_depth_abs_err_cmap'))
    os.mkdir(os.path.join(log_dir, 'gan'))
    #os.mkdir(args.output + "/feature")

    for iteration, data in enumerate(tqdm(val_loader)):
        img_L = data['img_L'].cuda()    # [bs, 1, H, W]
        img_R = data['img_R'].cuda()

        img_disp_l = data['img_disp_l'].cuda()
        img_depth_l = data['img_depth_l'].cuda()
        img_depth_realsense = data['img_depth_realsense'].cuda()
        img_label = data['img_label'].cuda()
        img_focal_length = data['focal_length'].cuda()
        img_baseline = data['baseline'].cuda()
        prefix = data['prefix'][0]

        img_disp_l = F.interpolate(img_disp_l, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_depth_l = F.interpolate(img_depth_l, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_depth_realsense = F.interpolate(img_depth_realsense, (540, 960), mode='nearest',
                             recompute_scale_factor=False)
        img_label = F.interpolate(img_label, (540, 960), mode='nearest',
                             recompute_scale_factor=False).type(torch.int)

        # If using warp_op, computing img_disp_l from img_disp_r
        if args.warp_op:
            img_disp_r = data['img_disp_r'].cuda()
            img_depth_r = data['img_depth_r'].cuda()
            img_disp_r = F.interpolate(img_disp_r, (540, 960), mode='nearest',
                                       recompute_scale_factor=False)
            img_depth_r = F.interpolate(img_depth_r, (540, 960), mode='nearest',
                                        recompute_scale_factor=False)
            img_disp_l = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))
            img_depth_l = apply_disparity_cu(img_depth_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]

        # If test on real dataset need to crop input image to (540, 960)

            

        #img_L_real = data['img_L_real'].cuda()    # [bs, 1, H, W]
        #img_L_real = F.interpolate(img_L_real, (540, 960), mode='bilinear',
        #                      recompute_scale_factor=False, align_corners=False)
        #img_L_o = F.interpolate(img_L, (540, 960), mode='bilinear',
                            #recompute_scale_factor=False, align_corners=False)
        #img_R_o = F.interpolate(img_R, (540, 960), mode='bilinear',
                            #recompute_scale_factor=False, align_corners=False)
        
        right_pad = cfg.REAL.PAD_WIDTH - 960
        top_pad = cfg.REAL.PAD_HEIGHT - 540
        img_L = F.pad(img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        img_R = F.pad(img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        

        # Save gan results
        #print(img_L_o.shape, img_R_o.shape, gan_model.real_A_L[:,0,:,:][:,None,:,:].shape, prefix)

        img_inputs = {
            'input': {
                'input_L': img_L, 'input_R': img_R
            }
        }
        #save_images_grid(summary_writer, 'test_gan', input_sample, iteration)
        save_images_grid(summary_writer, 'test_gan', img_inputs, iteration)

        # Pad the imput image and depth disp image to 960 * 544
        #right_pad = cfg.REAL.PAD_WIDTH - 960
        #top_pad = cfg.REAL.PAD_HEIGHT - 540
        #img_L = F.pad(img_L, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)
        #img_R = F.pad(img_R, (0, right_pad, top_pad, 0, 0, 0, 0, 0), mode='constant', value=0)

        if args.exclude_bg:
            print("bg flag is: ", args.exclude_bg)
            # Mask ground pixel to False
            img_ground_mask = (img_depth_l > 0) & (img_depth_l < 1.25)
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0) * img_ground_mask
        else:
            mask = (img_disp_l < cfg.ARGS.MAX_DISP) * (img_disp_l > 0)

        # Exclude uncertain pixel from realsense_depth_pred
        realsense_zeros_mask = img_depth_realsense > 0
        if args.exclude_zeros:
            print("zero flag is: ", args.exclude_zeros)
            mask = mask * realsense_zeros_mask
        mask = mask.type(torch.bool)

        ground_mask = torch.logical_not(mask).squeeze(0).squeeze(0).detach().cpu().numpy()

        with torch.no_grad():
            fake_L = G_AB(img_L)
            fake_R = G_AB(img_R)
            pred_disp = net(fake_L, fake_R)[0]

        pred_disp = pred_disp[:, :, top_pad:, :]  # TODO: if right_pad > 0 it needs to be (:-right_pad)
        pred_depth = img_focal_length * img_baseline / pred_disp  # pred depth in m

        # Get loss metric
        err_metrics = compute_err_metric(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                         img_baseline, mask)

        for k in total_err_metrics.keys():
            total_err_metrics[k] += err_metrics[k]
        logger.info(f'Test instance {prefix} - {err_metrics}')

        # Get object error
        obj_disp_err, obj_depth_err, obj_depth_4_err, obj_count = compute_obj_err(img_disp_l, img_depth_l, pred_disp, img_focal_length,
                                                     img_baseline, img_label, mask, cfg.SPLIT.OBJ_NUM)
        total_obj_disp_err += obj_disp_err
        total_obj_depth_err += obj_depth_err
        total_obj_depth_4_err += obj_depth_4_err
        total_obj_count += obj_count

        # Get disparity image
        pred_disp_np = pred_disp.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H, W]
        pred_disp_np[ground_mask] = -1
        pred_disp_np_o = torch.tensor(pred_disp_np)[None,:,:]

        # Get disparity ground truth image
        gt_disp_np = img_disp_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_disp_np[ground_mask] = -1
        gt_disp_np_o = torch.tensor(gt_disp_np)[None,:,:]

        # Get disparity error image
        pred_disp_err_np = disp_error_img(pred_disp, img_disp_l, mask)
        #print("disperr: ", pred_disp_err_np.shape)

        # Get depth image
        pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()  # in m, [H, W]
        # crop depth map to [0.2m, 2m]
        # pred_depth_np[pred_depth_np < 0.2] = -1
        # pred_depth_np[pred_depth_np > 2] = -1
        pred_depth_np[ground_mask] = -1
        pred_depth_np_o = torch.tensor(pred_depth_np)[None,:,:]

        # Get depth ground truth image
        gt_depth_np = img_depth_l.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_depth_np[ground_mask] = -1
        gt_depth_np_o = torch.tensor(gt_depth_np)[None,:,:]
        #print('LOG shape: ', img_depth_l.shape, gt_depth_np.shape)

        # Get depth error image
        pred_depth_err_np = depth_error_img(pred_depth * 1000, img_depth_l * 1000, mask)
        #print("deperr: ", pred_depth_err_np.shape)

        # Save images
        image_test_output = {'fake_L': fake_L, 'fake_R': fake_R, 'pred_disp': pred_disp_np_o, 'gt_disp': gt_disp_np_o, 'pred_depth': pred_depth_np_o, 'gt_depth': gt_depth_np_o}
        save_images(summary_writer, 'test_psmnet', image_test_output, iteration)
        save_img(log_dir, prefix, pred_disp_np, gt_disp_np, pred_disp_err_np,
                 pred_depth_np, gt_depth_np, pred_depth_err_np)

    # Get final error metrics
    for k in total_err_metrics.keys():
        total_err_metrics[k] /= len(val_loader)
    logger.info(f'\nTest on {len(val_loader)} instances\n {total_err_metrics}')

    # Save object error to csv file
    total_obj_disp_err /= total_obj_count
    total_obj_depth_err /= total_obj_count
    total_obj_depth_4_err /= total_obj_count
    save_obj_err_file(total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, log_dir)

    logger.info(f'Successfully saved object error to obj_err.txt')

    # Get error on real and 3d printed objects
    real_depth_error = 0
    real_depth_error_4mm = 0
    printed_depth_error = 0
    printed_depth_error_4mm = 0
    for i in range(cfg.SPLIT.OBJ_NUM):
        if i in real_obj_id:
            real_depth_error += total_obj_depth_err[i]
            real_depth_error_4mm += total_obj_depth_4_err[i]
        else:
            printed_depth_error += total_obj_depth_err[i]
            printed_depth_error_4mm += total_obj_depth_4_err[i]
    real_depth_error /= len(real_obj_id)
    real_depth_error_4mm /= len(real_obj_id)
    printed_depth_error /= (cfg.SPLIT.OBJ_NUM - len(real_obj_id))
    printed_depth_error_4mm /= (cfg.SPLIT.OBJ_NUM - len(real_obj_id))

    logger.info(f'Real objects - absolute depth error: {real_depth_error}, depth 4mm: {real_depth_error_4mm} \n'
                f'3D printed objects - absolute depth error {printed_depth_error}, depth 4mm: {printed_depth_error_4mm}')

    



def test(args,cfg):
    writer = SummaryWriter(args.output)

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
    G_AB = GeneratorResNet(input_shape, 2)
    G_BA = GeneratorResNet(input_shape, 2)
    D_A = Discriminator(1)
    D_B = Discriminator(1)

    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_dispnet_path:
                net = load_multi_gpu_checkpoint(net, args.load_dispnet_path, 'model')
            else:
                net.apply(weights_init_normal)
            G_AB = load_multi_gpu_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_multi_gpu_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_multi_gpu_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_multi_gpu_checkpoint(D_B, args.load_gan_path, 'D_B')
        else:
            if args.load_dispnet_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
            G_AB = load_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_checkpoint(D_B, args.load_gan_path, 'D_B')
    else:
        print('do not have pretrained model')
        return

    # optimizer = optim.SGD(params, momentum=0.9)

    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))
        G_AB = nn.DataParallel(G_AB, device_ids=list(range(args.use_multi_gpu)))
        G_BA = nn.DataParallel(G_BA, device_ids=list(range(args.use_multi_gpu)))
        D_A = nn.DataParallel(D_A, device_ids=list(range(args.use_multi_gpu)))
        D_B = nn.DataParallel(D_B, device_ids=list(range(args.use_multi_gpu)))

    net.to(device)
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)


    # data loader
    val_dataset = MessytableTestDataset_TEST(cfg.REAL.TRAIN, debug=False, sub=100, onReal=args.onReal)

    ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)
   

    print('begin testing...')
    
    # Tensorboard and logger
    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, f'{get_time_string()}_{args.annotate}')
    os.mkdir(log_dir)
    logger = setup_logger("CycleGAN-PSMNet Testing", distributed_rank=0, save_dir=log_dir)
    logger.info(f'Annotation: {args.annotate}')
    logger.info(f'Input args {args}')
    logger.info(f'Loaded config file \'{args.config_file}\'')
    logger.info(f'Running with configs:\n{cfg}')

    with torch.no_grad():
        if not args.onReal:
            test_sim(net, G_AB, G_BA, ValImgLoader, logger, log_dir, writer, args, cfg)
        else:
            test_sample(net, ValImgLoader, logger, log_dir, writer, args, cfg)

    #print('epoch:{}, D1:{:.6f}, EPE:{:.6f}'.format(epoch, D1, EPE))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    # training
    parser.add_argument('--maxdisp', type=int, default=192)

    # load & save checkpoints
    parser.add_argument('--load_checkpoints', nargs='?', type=int, default=0, help='load from ckp(saved by Pytorch)')
    parser.add_argument('--load_from_mgpus_model', nargs='?', type=int, default=0, help='load ckp which is saved by mgus(nn.DataParallel)')
    parser.add_argument('--load_gan_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--load_dispnet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')

    # tensorboard, print freq
    parser.add_argument('--print_freq', '-p', default=150, type=int, metavar='N', help='print frequency (default: 150)')

    # other
    parser.add_argument('--use_multi_gpu', nargs='?', type=int, default=0, help='the number of multi gpu to use')
    parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                    metavar='FILE', help='Config files')

    parser.add_argument('--output', type=str, default='../testing_output_cyclegan_psmnet', help='Path to output folder')
    parser.add_argument('--annotate', type=str, default='', help='Annotation to the experiment')
    parser.add_argument('--analyze-objects', action='store_true', default=True, help='Analyze on different objects')
    parser.add_argument('--exclude-bg', action='store_true', default=False, help='Exclude background when testing')
    parser.add_argument('--warp-op', action='store_true', default=True, help='Use warp_op function to get disparity')
    parser.add_argument('--exclude-zeros', action='store_true', default=False, help='Whether exclude zero pixels in realsense')
    parser.add_argument('--onReal', action='store_true', default=False)

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    #set_random_seed(args.seed)
    test(args,cfg)
