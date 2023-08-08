import torch

# setting up device agnostics
class EnvironmentSetup():
    def __init__(self):
        !pip install nni torch torchvision pytorch-lightning
    
    def set_device_agnostics():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device