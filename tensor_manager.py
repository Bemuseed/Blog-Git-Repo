import torch
import numpy

def get_tensors(dct):
    tensors = list(dct['state_dict'].values())
    new_dict = {'state_dict': dict()}
    for k in list(dct['state_dict'].keys()):
        new_dict['state_dict'][k] = 0
    return tensors, new_dict

def restore_tensor_dict(tensors, dict_template):
    t_counter = 0
    new_dict = {'state_dict': dict()}
    for k in list(dict_template['state_dict'].keys()):
       new_dict['state_dict'][k] = tensors[t_counter]
       t_counter += 1
    return new_dict


def get_tensors_as_lists(d):
    tensors, dict_template = get_tensors(d)
    return [t.numpy() for t in tensors], dict_template

def get_tensors_from_file(ckpt_file):
    tensor_dict = torch.load(ckpt_file, map_location="cpu")
    return get_tensors_as_lists(tensor_dict)
