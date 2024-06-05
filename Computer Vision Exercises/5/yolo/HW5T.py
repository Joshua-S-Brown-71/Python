"""
CS 4391 Homework 5 Programming
Implement the __getitem__() function in this python script
"""
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set='train', data_path='/Users/JOSH/Desktop/CS 4391                     (Vision)/HW/5/yolo/data'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):
    
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))
        
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]
        
        return gt_files_train, gt_files_val


    # TODO: implement this function
    def __getitem__(self, idx):
        # gt file
        filename_gt = self.gt_paths[idx]

        # Load image
        filename_image = os.path.splitext(filename_gt)[0].replace("-box", "") + '.jpg'
        image = cv2.imread(filename_image)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Unable to read image - {filename_image}")
            return {'image': torch.zeros((3, self.yolo_image_size, self.yolo_image_size)),
                    'gt_box': torch.zeros((5, self.yolo_grid_num, self.yolo_grid_num)),
                    'gt_mask': torch.zeros((self.yolo_grid_num, self.yolo_grid_num))}

        
    

        # Resize and normalize image
        image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))
        image = image.astype(np.float32) - self.pixel_mean
        image /= 255.0
        image = image.transpose((2, 0, 1))  # (channel, height, width)
        # Load ground truth bounding boxes
        with open(filename_gt, 'r') as f:
            gt_boxes = []
            for line in f:
                x1, y1, x2, y2 = map(float, line.strip().split())
                # Scale bounding box
                x1 *= self.scale_width
                y1 *= self.scale_height
                x2 *= self.scale_width
                y2 *= self.scale_height
                # Normalize bounding box
                cx = (x1 + x2) / 2 / self.yolo_image_size
                cy = (y1 + y2) / 2 / self.yolo_image_size
                w = (x2 - x1) / self.yolo_image_size
                h = (y2 - y1) / self.yolo_image_size
                gt_boxes.append([cx, cy, w, h, 1.0])  # Confidence is 1 for ground truth

        # Create gt_box tensor
        gt_box_blob = torch.zeros((5, self.yolo_grid_num, self.yolo_grid_num))
        for gt_box in gt_boxes:
            cx, cy, _, _, _ = gt_box
            grid_x = int(cx * self.yolo_grid_num)
            grid_y = int(cy * self.yolo_grid_num)
            gt_box_blob[:, grid_y, grid_x] = torch.tensor(gt_box)

        # Create gt_mask tensor
        gt_mask_blob = torch.zeros((self.yolo_grid_num, self.yolo_grid_num))
        for gt_box in gt_boxes:
            cx, cy, _, _, _ = gt_box
            grid_x = int(cx * self.yolo_grid_num)
            grid_y = int(cy * self.yolo_grid_num)
            gt_mask_blob[grid_y, grid_x] = 1.0

        # This is the sample dictionary to be returned from this function
        sample = {'image': torch.tensor(image),
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample


    # len of the dataset
    def __len__(self):
        return self.size
        

# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # visualize the training data
    for i, sample in enumerate(train_loader):
        
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print("\n",image.shape, gt_box.shape)
        
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize = 5.5)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=5.5)
        
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=5.5)
        plt.show()


"""
CS 4391 Homework 5 Programming
Implement the create_modules() function in this python script
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np


# the YOLO network class
class YOLO(nn.Module):
    def __init__(self, num_boxes, num_classes):
        super(YOLO, self).__init__()
        # number of bounding boxes per cell (2 in our case)
        self.num_boxes = num_boxes
        # number of classes for detection (1 in our case: cracker box)
        self.num_classes = num_classes
        self.image_size = 448
        self.grid_size = 64
        # create the network
        self.network = self.create_modules()
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    #TODO: implement this function to build the network
    def create_modules(self):
        modules = nn.Sequential()

        ### ADD YOUR CODE HERE ###
        # hint: use the modules.add_module()
          # Initial convolution layers
        modules.add_module('conv1', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu1', nn.ReLU())
        modules.add_module('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu2', nn.ReLU())
        modules.add_module('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu3', nn.ReLU())
        modules.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv4', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu4', nn.ReLU())
        modules.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv5', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu5', nn.ReLU())
        modules.add_module('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv6', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu6', nn.ReLU())
        modules.add_module('maxpool6', nn.MaxPool2d(kernel_size=2, stride=2))

        modules.add_module('conv7', nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu7', nn.ReLU())

        modules.add_module('conv8', nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu8', nn.ReLU())

        modules.add_module('conv9', nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        modules.add_module('relu9', nn.ReLU())

        # Flatten
        modules.add_module('flatten', nn.Flatten())

        # Fully connected layers
        modules.add_module('fc1', nn.Linear(50176, 256))
        modules.add_module('fc2', nn.Linear(256, 256))
        modules.add_module('fc_output', nn.Linear(256, 7 * 7 * (5 * self.num_boxes + self.num_classes)))
        modules.add_module('sigmoid', nn.Sigmoid())
        return modules


    # output (batch_size, 5*B + C, 7, 7)
    # In the network output (cx, cy, w, h) are normalized to be [0, 1]
    # This function undo the noramlization to obtain the bounding boxes in the orignial image space
    def transform_predictions(self, output):
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7)
        y = torch.linspace(0, 384, steps=7)
        corner_x, corner_y = torch.meshgrid(x, y, indexing='xy')
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        # corners are top-left corners for each cell in the grid
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pred_box = output.clone()

        # for each bounding box
        for i in range(self.num_boxes):
            # x and y
            pred_box[:, i*5, :, :] = corners[:, 0, :, :] + output[:, i*5, :, :] * self.grid_size
            pred_box[:, i*5+1, :, :] = corners[:, 1, :, :] + output[:, i*5+1, :, :] * self.grid_size
            # w and h
            pred_box[:, i*5+2, :, :] = output[:, i*5+2, :, :] * self.image_size
            pred_box[:, i*5+3, :, :] = output[:, i*5+3, :, :] * self.image_size

        return pred_box


    # forward pass of the YOLO network
    def forward(self, x):
        # raw output from the network
        output = self.network(x).reshape((-1, self.num_boxes * 5 + self.num_classes, 7, 7))
        # compute bounding boxes in the original image space
        pred_box = self.transform_predictions(output)
        return output, pred_box


# run this main function for testing
if __name__ == '__main__':
    network = YOLO(num_boxes=2, num_classes=1)
    print(network)

    image = np.random.uniform(-0.5, 0.5, size=(1, 3, 448, 448)).astype(np.float32)
    image_tensor = torch.from_numpy(image)
    print('input image:', image_tensor.shape)

    output, pred_box = network(image_tensor)
    print('network output:', output.shape, pred_box.shape)
    
    
"""
CS 4391 Homework 5 Programming
Implement the compute_loss() function in this python script
"""
import os
import torch
import torch.nn as nn


# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
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
    iou = inter / (areap + areag - inter)    
    return iou

# TODO: finish the implementation of this loss function for YOLO training
# output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# num_boxes: number of bounding boxes per cell
# num_classes: number of object classes for detection
# grid_size: YOLO grid size, 64 in our case
# image_size: YOLO image size, 448 in our case
def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    num_output = num_boxes * 5 + num_classes

    # compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    # compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                # if the gt mask is 1
                if gt_mask[i, j, k] > 0:
                    # transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * (image_size / num_grids)
                    gt[1] = gt[1] * grid_size + j * (image_size / num_grids)
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1

                    # select the one with maximum IoU
                    for b in range(num_boxes):
                        # center x, y and width, height
                        pred = pred_box[i, 5 * b:5 * b + 4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b

                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou

    # compute yolo loss
    weight_coord = 5.0
    weight_noobj = 0.5

    loss_x = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 0::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))

    loss_y = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 1::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))
    loss_w = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 2::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))
    loss_h = torch.sum(box_mask * weight_coord * torch.pow(pred_box[:, 3::num_output, :, :] - gt_box[:, 0:1, :, :], 2.0))



    # loss function on confidence for objects
    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4::num_output, :, :], 2.0))

    # loss function on confidence for non-objects
    loss_noobj = torch.sum((1 - box_mask) * torch.pow(box_confidence - output[:, 4::num_output, :, :], 2.0))

    # loss function for object class
    loss_cls = torch.sum(box_mask * torch.pow(output[:, 5*num_boxes:, :, :], 2.0))

    # total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + weight_noobj * loss_noobj + loss_cls

    return loss
