import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
import visdom

class Visualizer():
    """ This class can includes severasl functions that can display images and print logging information."""
    def __init__(self, configuration):
        """ Initialize the Visualizer class.
        
        Input params:
            configuration -- stores all the configuration"""
        self.configuration = configuration
        self.display_id = 0
        self.name = configurationp['name']
        
        self.ncols = 0
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            self.create_visdom_connections()
    
    def reset(self):
        pass
    
    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server'
        print('\nCould not connect to Visdom server. \Trying to start a new server ...')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        
    def plot_current_losses(self, epoch, counter_ratio, losses):
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        y = np.sqeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X = x,
                Y = y,
                opts={
                    'title' : self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win = self.display_id)
        except ConnectionError:
            self.create_visdom_connections()
            
    def plot_current_validation_metrics(self, epoch, metrics):
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.sqeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts = {
                    'tile': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1
            )
        except ConnectionError:
            self.create_visdom_connections()
            
    def plot_roc_curve(self, fpr, tpr, thresholds):
        try:
            self.vis.line(
                X=fpr,
                Y=tpr,
                opts={
                    'title': 'ROC curve',
                    'xlabel': '1 - Specificity',
                    'ylabel': 'Sensitivity',
                    'fillarea': True},
                win=self.display_id+2
            )
        except ConnectionError:
            self.create_visdom_connections()
    
    def show_validation_images(self, images):
        images = images.permute(1, 0, 2 , 3)
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
        
        images = images[:, None, :, :]
        
        try:
            self.vision.images(images, win=self.display_id+3, nrow=3)
        except ConnectionError:
            self.create_visdom_connections()
            
    def print_current_losses(self, eopch, max_epochs, iter, max_iters, losses):
        message = '[Epoch: {}/{}, Iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
        for k, v in losses.items():
            message += '{}: {.4f} '.format(k, v)
        print(message)