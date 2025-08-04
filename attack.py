import torch
import torch.nn as nn

import torchvision.utils
from robustbench.model_zoo.imagenet import imagenet_models
from robustbench.model_zoo.imagenet import linf
from robustbench.model_zoo.enums import BenchmarkDataset

class Attack(object):
    def __init__(self, attack, model_name, epsilon, loss, device=None):
        self.attack = attack
        self.model = self.load_model(model_name)
        self.epsilon = epsilon
        self.device = self.model.device
        self.loss = nn.CrossEntropyLoss()
        self.model_name = model_name

    def load_model(self, model_name):
           adversarial_keys = [inner_key for outer_key, inner_dict in imagenet_models.items() for inner_key in inner_dict.keys()]
            if model_name in models.__dict__.keys():
                print('=> Loading model {} from torchvision.models'.format(model_name))
                model = models.__dict__[model_name](pretrained=True)
            elif model_name in timm.list_models():
                print('=> Loading model {} from timm.models'.format(model_name))
                model = timm.create_model(model_name, pretrained=True)
            elif model_name in adversarial_keys:
                print('=> Loading model {} from adversarial_models'.format(model_name))
                model = load_model(model_name=model_name, threat_model='Linf', dataset=BenchmarkDataset.imagenet)
            elif model_name not in list(linf.keys()):
                print('=> Loading model {} from adversarial_models'.format(model_name))
                model = adv_model_dict[model_name]()
                checkpoint = torch.load(f'./models/imagenet/Linf/{model_name}.pth')
                model.load_state_dict(checkpoint)
            return model.eval().cuda()

    def forward(self, data, label, filenames, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        adv_data = data.clone().detach()
        accumulated_grad = torch.zeros_like(data).to(self.device)
        
        delta = self.init_delta(data)
        for step in range(steps):
                adv_data.requires_grad = True
                outputs = self.model(adv_data)
                loss = self.criterion(outputs, label)
                
                self.model.zero_grad()
                loss.backward()
                grad = adv_data.grad.detach()

                # saliency_map(grad, filenames)

                self.attack(grad)

                adv_data = adv_data + alpha * grad.sign()
                perturbation = torch.clamp(adv_data - data, min=-eps, max=eps)
                adv_data = torch.clamp(data + perturbation, 0, 1).detach()

       return adv_data
