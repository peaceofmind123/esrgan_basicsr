# experiment 1: run ESRGAN with pretrained weights
from basicsr.models.esrgan_model import ESRGANModel
import yaml
import os

current_dir = os.getcwd()
print(current_dir)
with open(current_dir + "/basicsr/options/test/ESRGAN/experiment_01_ESRGAN.yml", "r") as stream:
    try:
        opt = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
model = ESRGANModel(opt)
print(model.net_g.children)
