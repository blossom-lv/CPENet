import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./model')
import numpy as np
from datetime import datetime
from model.vit_iterate import vit_iterate
from model.swin_iterate import swin_iterate
from model.vit_attentiona import vit_attentiona
from model.vit_attentionb import vit_attentionb
from model.vit_attentionc import vit_attentionc
from model.vitxxs_attention import vitxxs_attention
from model.swin_attentiona import swin_attentiona
from model.swin_attentionb import swin_attentionb
from model.swin_attentionc import swin_attentionc
from model.ablation_modules.base import base
from model.ablation_modules.add_SIEM import add_SIEM
from model.ablation_modules.add_SIEM_PDMA import add_SIEM_PDMA
from model.PDMA.wo_CrA import wo_CrA
from model.idpdecoder.idp_decoder import idp_decoder
#from model.smt_attention import smt_attention
#from model.smtt_attention import smtt_attention
from torchvision.utils import make_grid
from data.data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from regionloss import masked_bce_with_logits_loss, iou_region_loss


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    mae = checkpoint['mae']
    best_epoch = checkpoint['best_epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}, best_mae {mae:.4f} of best_epoch {best_epoch}")
    return model, optimizer, epoch, mae, best_epoch

def weights_decay_50(weights):
    for i in range(len(weights)):
        weights[i] *= 0.9
    return weights

if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
masks_root = opt.masks_root#add

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path
input_size = opt.trainsize

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + 'Train.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Train")
model = vit_attentiona(input_size)#input_size

num_parms = 0
if (opt.load_pre is not None):
    #model.load_pre(opt.load_pre, './mobilevit_s.pt')
    model.load_pre(opt.load_pre)
    print('load model from ', opt.load_pre)

# freeze the backbone
for param in model.rgb_vit_iterate.parameters() and model.depth_vit_iterate.parameters():# swin vit_iterate
    param.requires_grad = False

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)#part frozen optimizer

# 尝试加载之前保存的检查点
try:
    breakpoint_train = os.path.join(save_path, opt.load_point)
    model, optimizer, start_epoch, best_mae, best_epoch = load_checkpoint(model, optimizer, breakpoint_train)
except FileNotFoundError:
    start_epoch = 0
    best_mae = 1
    best_epoch = 0

# load data
print('load data...')
#train_loader = get_loader(image_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
train_loader = get_loader(image_root, gt_root,depth_root, masks_root, batchsize=opt.batchsize, trainsize=opt.trainsize)#add_revise
test_loader = test_dataset(test_image_root, test_gt_root,test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_pre, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
#best_mae = 1
#best_epoch = 0
decay_num = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step, decay_num
    model.train()

    loss_all = 0
    epoch_step = 0
    decay_num += 1
    freeze_time = 0
    freeze_part = 0
    mask_weights = [0.1, 0.2, 0.3, 0.4] 
    iou_threshold = 0.5

    try:
        #for i, (images, gts, depth) in enumerate(train_loader, start=1):
        for i, (images, gts, depth, masks) in enumerate(train_loader, start=1):#add_revise
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1,3,1,1).cuda()
            masks = [torch.tensor(mask).float().cuda() for mask in masks]#add
            s1,s2,s3,s4 = model(images,depth)

            bce_iou1 = CE(s1, gts) + iou_loss(s1, gts)
            bce_iou2 = CE(s2, gts) + iou_loss(s2, gts)
            bce_iou3 = CE(s3, gts) + iou_loss(s3, gts)
            bce_iou4 = CE(s4, gts) + iou_loss(s4, gts)

            #if epoch >= 1:# 50: if iou of gt and pred is more than 0.5, use the region loss 区域损失函数
            if iou_loss(s1, gts) < iou_threshold and iou_loss(s2, gts) < iou_threshold and iou_loss(s3, gts) < iou_threshold and iou_loss(s4, gts) < iou_threshold:
                for j in range(4):#add
                    masks_loss_1 = mask_weights[j] * (masked_bce_with_logits_loss(s1, gts, masks[j]) + iou_region_loss(s1, masks[j], gts))
                    bce_iou1 -= masks_loss_1
                    masks_loss_2 = mask_weights[j] * (masked_bce_with_logits_loss(s2, gts, masks[j]) + iou_region_loss(s2, masks[j], gts))
                    bce_iou2 -= masks_loss_2
                    masks_loss_3 = mask_weights[j] * (masked_bce_with_logits_loss(s3, gts, masks[j]) + iou_region_loss(s3, masks[j], gts))
                    bce_iou3 -= masks_loss_3
                    masks_loss_4 = mask_weights[j] * (masked_bce_with_logits_loss(s4, gts, masks[j]) + iou_region_loss(s4, masks[j], gts))
                    bce_iou4 -= masks_loss_4

            bce_iou_deep_supervision = bce_iou1+bce_iou2+bce_iou3+bce_iou4

            loss = bce_iou_deep_supervision
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # freeze end 渐近冻结策略
            if 41 <= epoch and 0 == freeze_time:
                #print('\nbackbone not frozen')
                freeze_time = 1
                for param in model.rgb_vit_iterate.parameters() and model.depth_vit_iterate.parameters():# swin vit_iterate
                    param.requires_grad = True
                tmp_lr = optimizer.state_dict()['param_groups'][0]['lr']
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), tmp_lr)
            
            if 41 <= epoch <= 80 and 0 == freeze_part:
                freeze_part = 1
                model.rgb_vit_iterate.layer_1.trainable = True
                model.rgb_vit_iterate.layer_2.trainable = True
                model.rgb_vit_iterate.layer_3.trainable = False
                model.rgb_vit_iterate.layer_4.trainable = False
                model.rgb_vit_iterate.layer_5.trainable = False
            if 41 == epoch and 1 == freeze_part:
                freeze_part = 0
            if 80 < epoch and 0 == freeze_part:
                freeze_part = 1
                model.rgb_vit_iterate.layer_1.trainable = True
                model.rgb_vit_iterate.layer_2.trainable = True
                model.rgb_vit_iterate.layer_3.trainable = True
                model.rgb_vit_iterate.layer_4.trainable = True
                model.rgb_vit_iterate.layer_5.trainable = False
            if 120 == epoch and 1 == freeze_part:
                freeze_part = 0
            if 121 < epoch and 0 == freeze_part:
                freeze_part = 1
                model.rgb_vit_iterate.layer_1.trainable = True
                model.rgb_vit_iterate.layer_2.trainable = True
                model.rgb_vit_iterate.layer_3.trainable = True
                model.rgb_vit_iterate.layer_4.trainable = True
                model.rgb_vit_iterate.layer_5.trainable = True

            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s3[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch >= 180:
             torch.save(model.state_dict(), save_path + 'model_epoch_{}.pth'.format(epoch))
        #breakpoint training
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mae': best_mae,
        'best_epoch': best_epoch
    }
        torch.save(checkpoint, save_path + 'model_epoch_last.pth')
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'model_epoch_{}.pth'.format(epoch))
        print('save checkpoints successfully!')
        raise
# def bce2d_new(input, target, reduction=None):
#     assert (input.size() == target.size())
#     pos = torch.eq(target, 1).float()
#     neg = torch.eq(target, 0).float()

#     num_pos = torch.sum(pos)
#     num_neg = torch.sum(neg)
#     num_total = num_pos + num_neg

#     alpha = num_neg / num_total
#     beta = 1.1 * num_pos / num_total
#     weights = alpha * pos + beta * neg

#     return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1,3,1,1).cuda()
            res,res2,res3,res4 = model(image,depth)
            res = res+res2+res3+res4
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'model_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(start_epoch + 1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch > 180:
            test(test_loader, model, epoch, save_path)
