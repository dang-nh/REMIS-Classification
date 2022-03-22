from collections import OrderedDict
from models.abstract_model import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class SimpleClassificationModel(AbstractModel):
    def __init__(self, configuration):
        super().__init__(configuration)
        
        self.loss_names = ['classification']
        self.network_names = ['simple cnn']
        
        self.simplecnn = double_conv(3, 4)
        self.simplecnn = self.simplecnn.to(self.device)
        if self.is_train:
            self.criterion_loss = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.simplecnn.parameters(), lr=self.configuration['lr'])
            self.optimizers = [self.optimizer]
            
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []
        
    def forward(self):
        self.output = self.simplecnn(self.input)
        
    def backward(self):
        self.loss_classification = self.criterion_loss(self.output, self.label)
        
    def optimize_parameters(self):
        self.loss_classification.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        
    def test(self):
        super().test()
        
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)
        
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()
        
        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()
        
        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()
        
        val_accuracyu = accuracy_score(labels, predictions)
        
        metrics = OrderedDict()
        metrics['accuracy'] = val_accuracy
        
        visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {.3f}'.format(val_accuracy))
        
        self.val_images = []
        self.val_predictions = []
        self.val_labels = []