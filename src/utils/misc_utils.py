import torch
import os

def get_model_size(model_name, saved=True):
    if saved:
        return '{} MB'.format(os.path.getsize(model_name) / 1e6)
    else:
        torch.save(model_name.state_dict(), 'temp.pth')
        size_mb = os.path.getsize('temp.pth') / 1e6
        os.remove('temp.pth')
        return '{} MB'.format(size_mb)