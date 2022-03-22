import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

class AbstractModel(ABC):
    
    def _init__(self, configuration):
        self.configuration = configuration
        self.is_train = configuration['is_train']
        self.use_cude = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cude else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.save_dir = configuration['checkpoint_path']
        self.network_names = []
        self.loss_names = []
        self.optimizers = []
        self.visual_names = []
        
    def set_input(self, input):
        self.input = transfer_to_device(input[0], self.device)
        self.label = transfer_to_device(input[1], self.device)
        
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def optimize_parameters(self):
        pass
    
    def setup(self):
        if self.configuration['load_checkpoint'] >= 0:
            last_checkpoint = self.configuration['load_checkpoint']
        else:
            last_checkpoint = -1
            
        if last_checkpoint >= 0:
            self.load_networks(last_checkpoint)
            if self.is_train:
                self.load_optimizer(last_checkpoint)
                for o in self.optimizers:
                    o.param_groups[0]['lr'] = o.param_groups[0]['initial_lr']
        
        self.schedulers = [get_scheduler(optimizer, self.configuration) for optimizer in self.optimizers]
        
        if last_checkpoint > 0:
            for s in self.schedulers:
                for _ in range(last_checkpoint):
                    s.step()
                    
        self.print_networks()
        
    def train(self):
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
                
    def evel(self):
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
                
    def test(self):
        with torch.no_grad():
            self.forward()
            
    def updatte_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {%.7f}'.format(lr))
        
    def save_networks(self, epoch):
        for name in self.network_names:
            if isinstance(name, str):
                save_filename = '{}_net_{}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                
            if self.use_cuda:
                torch.save(net.cpu().state_dict(), save_path)
                net.to(self.device)
                
            else:
                torch.save(net.cpu().state_dict(), save_path)
    
    def load_networks(self, epoch):
        for name in self.network_names:
            if isinstance(name, str):
                load_filenames = '{}_net_{}.pth'.format(epoch, name)
                load_path = os.path.join(self.save_dir, load_filenames)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('Loading model from {}'.format(load_path))
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                    
                net.load_state_dict(state_dict)
                
    def save_optimizers(self, epoch):
        for i, optimizer in enumerate(self.optimizers):
            save_filename = '{}_optimizer_{}.pth'.format(epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(optimizer.state_dict(), save_path)
            
    def load_optimizers(Self, epoch):
        for i, optimizer in enumerate(self.optimizer):
            load_filenames = '{}_optimizer_{}.pth'.format(epoch, i)
            load_path = os.path.join(self.save_dir, load_filenames)
            print('Loading the optimizer from {}'.format(load_path))
            state_dict = torch.load(load_path)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)
            
    def print_neworks(self):
        print('Networks intialized')
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net + name')
                num_params = 0 
                for params in net.parameters():
                    num_params += params.numel()
                print(net)
                print('Network {}. Total number or parameters: {.3f} M'.format(name, num_params/1e6))
        
        
    def set_requires_grad(self, requires_grad = False):
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret
    
    def pre_epoch_callback(self, epoch):
        pass
    
    def post_epoch_callack(self, epoch, visualizer):
        pass
    
    def get_hyperparam_result(self):
        pass
    
    def export(self):
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                export_path = os.path.join(self.configuration['export_path'], 'exported_net_{}.pth'.format(name))
            traced_script_module = torch.jit.trace(net, self.input)
            traced_script_module.save(export_path)
            
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            visual_ret[name] = getattr(self, name)
        return visual_ret