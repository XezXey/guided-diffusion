import torch as th
import copy

def make_deepcopyable(model_kwargs, keys):
    '''
    Make the dict-like to be used with deepcopy function
    :param model_kwargs: a dict-like with {'key1': tensor, ...}
    :param keys: a keys of tensor that need to detach() first before to use a deepcopy
        - only 2 possible type : 1. tensor, 2. list-of-tensor
    :return dict_tensor: the deepcopy version of input dict_tensor
    '''

    for key in keys:
        if th.is_tensor(model_kwargs[key]):
            model_kwargs[key] = model_kwargs[key].detach()
        elif isinstance(model_kwargs[key], list):
            for i in range(len(model_kwargs[key])):
                model_kwargs[key][i] = model_kwargs[key][i].detach()

    model_kwargs_copy = copy.deepcopy(model_kwargs)
    return model_kwargs_copy

