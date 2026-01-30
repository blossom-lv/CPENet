import torch.nn as nn
import torch

def masked_bce_with_logits_loss(logits, target, mask):
    """
    计算在 mask 区域内的 BCEWithLogitsLoss。
    
    参数:
    logits (Tensor): 模型的原始输出，形状为 (batch_size, height, width)
    target (Tensor): 真实标签，形状为 (batch_size, height, width)
    mask (Tensor): 区域掩码，形状为 (batch_size, height, width)，1 表示计算损失，0 表示忽略损失
    
    返回:
    Tensor: 计算的损失值
    """
    # 定义 BCEWithLogitsLoss，使用 'none' 逐像素计算损失
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    # 计算每个像素的损失
    loss = loss_fn(logits, target)
    #print('loss_ce_in', loss.mean())
    
    # 将损失与 mask 相乘，屏蔽不需要计算损失的区域
    masked_loss = loss * mask
    #print('loss_mask_in', masked_loss.mean())
    #add of thinking: masked_loss * newmask in which the newmask is obtianed by set the region where intensity of pixels  not over 0.2 of the pred
    # edge_mask = logits * (logits < 0.2).float()
    # masked_loss *= edge_mask
    
    return masked_loss.mean()

def iou_region_loss(pred, mask, gt):
    total_loss = iou_loss(pred, gt)

    #test_new_thinking
    # edge_mask = pred * (pred < 0.2).float()
    # mask_1 = edge_mask * mask

    inverted_mask = 1 - mask#change needed of thinking
    pred_masked = pred * inverted_mask
    gt_masked = gt * inverted_mask
    loss_masked = iou_loss(pred_masked, gt_masked)
    return max(total_loss - loss_masked, 0)

    #add of thinking: total is still the same, mask need to be changed to the overlap of the old mask and the not over 0.2 region of pred

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()