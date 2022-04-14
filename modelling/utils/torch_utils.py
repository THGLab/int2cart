import numpy as np

def tensor_to_numpy(tensor):
    if type(tensor) is np.ndarray:
        return tensor
    return  tensor.detach().cpu().numpy()

def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        delattr(curr_mod, name)
        setattr(curr_mod, name, param)

def set_model_parameters(model, parameter_dict):
    for item in parameter_dict:
        set_param(model, item, parameter_dict[item])