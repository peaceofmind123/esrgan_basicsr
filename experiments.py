# experiment 1: run ESRGAN with pretrained weights
from basicsr.models.esrgan_model import ESRGANModel
import yaml
import os
import torch

current_dir = os.getcwd()
print(current_dir)
with open(current_dir + "/basicsr/options/test/ESRGAN/experiment_01_ESRGAN.yml", "r") as stream:
    try:
        opt = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
model = ESRGANModel(opt)


class MyModel(torch.nn.Module):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self._model = model
        self.features = torch.nn.Sequential(
            *list(model.net_g.children())[:3]
        )

    def forward(self, x):
        y = self.features(x)
        return y

myModel = MyModel(model)
arr = torch.randn((3,64))
print(myModel(arr))