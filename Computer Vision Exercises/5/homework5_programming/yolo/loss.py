import os
import torch
import torch.nn as nn

def compute_iou(pred, gt):
    # Compute Intersection over Union (IoU) between predicted and ground truth boxes
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    if areap + areag - inter == 0:
        iou = 0
    else:
        iou = inter / (areap + areag - inter)

    return iou

def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    # Compute the YOLO loss
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    num_output = num_boxes * 5 + num_classes

    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                if gt_mask[i, j, k] > 0:
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * (image_size / num_grids)
                    gt[1] = gt[1] * grid_size + j * (image_size / num_grids)
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1

                    for b in range(num_boxes):
                        pred = pred_box[i, 5 * b:5 * b + 4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b

                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou

    weight_coord = 5.0
    weight_noobj = 0.5

    # Calculate individual loss components
    loss_x = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 0::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))
    loss_y = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 1::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))
    loss_w = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 2::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))
    loss_h = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 3::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))

    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4::num_output, :, :], 2.0))
    loss_noobj = torch.sum((1 - box_mask) * torch.pow(box_confidence - output[:, 4::num_output, :, :], 2.0))
    loss_cls = torch.sum(box_mask * torch.pow(output[:, 5 * num_boxes:, :, :], 2.0))

    # Compute total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + weight_noobj * loss_noobj + loss_cls

    return loss
