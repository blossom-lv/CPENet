import torch
import torch.nn.functional as F
import sys
sys.path.append('./model')
import numpy as np
import os, argparse
import cv2
#from model.vit_iterate96 import vit_iterate
from model.vit_iterate import vit_iterate
from model.vit_attention import vit_attention
from model.vit_attentiona import vit_attentiona
from model.vit_attentionb import vit_attentionb
from model.vit_attentionc import vit_attentionc
from model.vit_attentiond import vit_attentiond
from model.vitxxs_attention import vitxxs_attention
from model.swin_attentionb import swin_attentionb
from model.ablation_modules.base import base
from model.ablation_modules.add_SIEM import add_SIEM
from model.ablation_modules.add_SIEM_PDMA import add_SIEM_PDMA
from model.SIEM.wo_AtM import wo_AtM
from model.SIEM.wo_SIEM import wo_SIEM
from model.PDMA.wo_CrA import wo_CrA
from model.idpdecoder.idp_decoder import idp_decoder
#from model.asym_attention import asym_attention
#from model.asym_attentiona import asym_attentiona
from data.data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../CPNet-main/dataset/test_datasets/',help='test dataset path')# test_datasets train_dut
opt = parser.parse_args()

dataset_path = opt.test_path
test_size = opt.testsize

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

def load_checkpoint(model, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

#load the model
model = vit_attentiona(test_size)
model.load_state_dict(torch.load('./exp_learning/model_epoch_best.pth'), strict=False)#156 165 172 exp_ablation_1234
# model = load_checkpoint(model, './exp_va_freeze_iou5_models_w_region_loss/model_epoch_last.pth')
model.cuda()
model.eval()

test_datasets = ['DUT-RGBD', 'SSD', 'ReDWeb', 'DES', 'LFSD', 'NJU2K', 'NLPR', 'SIP', 'STERE']#'COME-E', 'COME-H'
# test_datasets = ['COME-H', 'COME-E']

for dataset in test_datasets:
    save_path = './test_maps_ablation_idp_decoder/' + dataset + '/'#_new_loss_abs_continue
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    mae_sum = 0
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1,3,1,1).cuda()
        res, res2, res3, res4 = model(image,depth)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        #print('save img to: ',save_path+name)
        # if dataset == 'COME-H':
        cv2.imwrite(save_path + name, res*255)
    mae = mae_sum / test_loader.size
    print('MAE: {} of dataset {}\n'.format(mae, dataset))
