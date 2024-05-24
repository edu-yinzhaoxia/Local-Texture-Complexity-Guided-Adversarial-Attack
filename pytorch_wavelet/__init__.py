import torch
from . import DWT

def wavelet_mse(ori, adv):
    ori_dwt = DWT.DWT_2D("haar")
    adv_dwt = DWT.DWT_2D("haar")
    ori_ll, _, _, _ = ori_dwt(ori)
    adv_ll, _, _, _ = adv_dwt(adv)
    return torch.nn.functional.mse_loss(ori_ll, adv_ll)
