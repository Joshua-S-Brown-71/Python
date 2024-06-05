import os
import math
import torch
import torch.nn as nn
import numpy as np

class YOLO(nn.Module):
    def __init__(self, num_boxes, num_classes):
        super(YOLO, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.image_size = 448
        self.grid_size = 64
        self.network = self.create_modules()

        # Initialize weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def create_modules(self):
        # Create a list to hold the layers of the YOLO network
        modules = nn.ModuleList()

        # Convolutional layers
        modules.extend([
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten()
        ])

        # Fully connected layers
        modules.extend([
            nn.Linear(50176, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7 * 7 * (5 * self.num_boxes + self.num_classes)),
            nn.Sigmoid()
        ])

        return nn.ModuleList(modules)

    def transform_predictions(self, output):
        # Function to transform network output to bounding box predictions
        batch_size = output.shape[0]
        x = torch.linspace(0, 384, steps=7)
        y = torch.linspace(0, 384, steps=7)
        corner_x, corner_y = torch.meshgrid(x, y, indexing='xy')
        corner_x = torch.unsqueeze(corner_x, dim=0)
        corner_y = torch.unsqueeze(corner_y, dim=0)
        corners = torch.cat((corner_x, corner_y), dim=0)
        corners = corners.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        pred_box = output.clone()

        for i in range(self.num_boxes):
            pred_box[:, i*5, :, :] = corners[:, 0, :, :] + output[:, i*5, :, :] * self.grid_size
            pred_box[:, i*5+1, :, :] = corners[:, 1, :, :] + output[:, i*5+1, :, :] * self.grid_size
            pred_box[:, i*5+2, :, :] = output[:, i*5+2, :, :] * self.image_size
            pred_box[:, i*5+3, :, :] = output[:, i*5+3, :, :] * self.image_size

        return pred_box
    
    def forward(self, x):
        # Forward pass through the YOLO network
        outputs = {}
        for module in self.network:
            x = module(x)

        outputs['fc_output'] = x.reshape((-1, self.num_boxes * 5 + self.num_classes, 7, 7))
        outputs['pred_box'] = self.transform_predictions(outputs['fc_output'])
        return outputs

if __name__ == '__main__':
    # Sample usage of the YOLO model
    network = YOLO(num_boxes=2, num_classes=1)
    print(network)

    image = np.random.uniform(-0.5, 0.5, size=(1, 3, 448, 448)).astype(np.float32)
    image_tensor = torch.from_numpy(image)
    print('input image:', image_tensor.shape)

    output, pred_box = network(image_tensor)
    print('network output:', output.shape, pred_box.shape)
