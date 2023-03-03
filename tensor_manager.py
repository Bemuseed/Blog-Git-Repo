import torch
import numpy

def get_tensors(dct):
    tensors = list(dct['state_dict'].values())
    new_dict = {'state_dict': dict()}
    for k in list(dct['state_dict'].keys()):
        new_dict['state_dict'][k] = 0
    return [t.numpy() for t in tensors], new_dict

def restore_tensor_dict(tensors, dict_template):
    t_counter = 0
    new_dict = {'state_dict': dict()}
    for k in list(dict_template['state_dict'].keys()):
       new_dict['state_dict'][k] = torch.from_numpy(tensors[t_counter])
       t_counter += 1
    return new_dict

def get_tensors_from_file(ckpt_file):
    tensor_dict = torch.load(ckpt_file, map_location="cpu")
    return get_tensors(tensor_dict)
