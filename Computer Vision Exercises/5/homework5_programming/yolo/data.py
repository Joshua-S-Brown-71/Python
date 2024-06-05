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

class CrackerBox(data.Dataset):
    def __init__(self, image_set='train', data_path='data'):
        # Initialize the dataset with specified image set (train or val) and data path
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
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)

    def list_dataset(self):
        # List the dataset by reading ground truth files
        filename = os.path.join(self.data_path, '*.txt')
        gt_files = sorted(glob.glob(filename))
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]
        return gt_files_train, gt_files_val

    def __getitem__(self, idx):
        # Get item at index 'idx' from the dataset
        filename_gt = self.gt_paths[idx]
        filename_image = os.path.splitext(filename_gt)[0].replace("-box", "") + '.jpg'
        image = cv2.imread(filename_image)
        
        # Check if image loading is successful
        if image is None:
            print(f"Error: Unable to read image - {filename_image}")
            return {'image': torch.zeros((3, self.yolo_image_size, self.yolo_image_size)),
                    'gt_box': torch.zeros((5, self.yolo_grid_num, self.yolo_grid_num)),
                    'gt_mask': torch.zeros((self.yolo_grid_num, self.yolo_grid_num))}
        
        # Resize and preprocess the image
        image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))
        image = image.astype(np.float32) - self.pixel_mean
        image /= 255.0
        image = image.transpose((2, 0, 1))
        
        # Read ground truth bounding boxes from the annotation file
        with open(filename_gt, 'r') as f:
            gt_boxes = []
            for line in f:
                x1, y1, x2, y2 = map(float, line.strip().split())
                x1 *= self.scale_width
                y1 *= self.scale_height
                x2 *= self.scale_width
                y2 *= self.scale_height
                cx = (x1 + x2) / 2 / self.yolo_image_size
                cy = (y1 + y2) / 2 / self.yolo_image_size
                w = (x2 - x1) / self.yolo_image_size
                h = (y2 - y1) / self.yolo_image_size
                gt_boxes.append([cx, cy, w, h, 1.0])

        # Create tensors for ground truth bounding boxes and masks
        gt_box_blob = torch.zeros((5, self.yolo_grid_num, self.yolo_grid_num))
        for gt_box in gt_boxes:
            cx, cy, _, _, _ = gt_box
            grid_x = int(cx * self.yolo_grid_num)
            grid_y = int(cy * self.yolo_grid_num)
            gt_box_blob[:, grid_y, grid_x] = torch.tensor(gt_box)

        gt_mask_blob = torch.zeros((self.yolo_grid_num, self.yolo_grid_num))
        for gt_box in gt_boxes:
            cx, cy, _, _, _ = gt_box
            grid_x = int(cx * self.yolo_grid_num)
            grid_y = int(cy * self.yolo_grid_num)
            gt_mask_blob[grid_y, grid_x] = 1.0

        # Return the sample as a dictionary
        sample = {'image': torch.tensor(image),
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}
        return sample

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.size

def draw_grid(image, line_space=64):
    # Draw a grid on the image for visualization
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]

if __name__ == '__main__':
    # Example usage of the CrackerBox dataset
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # Iterate over the dataset and visualize samples
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
        
        # Display the sample
        print("\n", image.shape, gt_box.shape)
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize=5.5)
        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=5.5)
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=5.5)
        plt.show()
