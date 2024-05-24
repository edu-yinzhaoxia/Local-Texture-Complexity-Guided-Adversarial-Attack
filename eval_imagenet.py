import torch
from torchvision import models
from torch import nn
from tqdm import tqdm
from torchattacks import PGD

from utils.auxiliary_utils import *
from utils.eval_metric_utils import *
from LTCA import LTCA as LTCA


def main():
    alpha = 0.05
    epsilon = 0.015
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1

    data_dir = "../datasets/picked_data"
    json_file = "./imagenet_class_index.json"
    _, test_loader, _ = load_data(data_dir, json_file, batch_size)
    
    
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resnet_model = nn.Sequential(
        norm_layer,
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    ).to(device)
    resnet_model = resnet_model.eval()

    # loader = tqdm(test_loader)
    attack = LTCA(model=resnet_model, image_size=224, device=device, alpha=alpha, epsilon=epsilon)
    PerD = PerceptualDistance("haar")
    
    torch.manual_seed(1)
    
    success_cnt, total_present = 0, 0
    
    with tqdm(test_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            x_adv = attack(x, y)
            
            # torchvision.utils.save_image(x_adv, "./temp_adv.png")
            # torchvision.utils.save_image(x, "./temp_ori.png")
            
            with torch.no_grad():
                pred = resnet_model(x_adv).argmax(dim=1)
                
            success_index = pred != y
            temp_cnt = sum(success_index).item()
            success_cnt += temp_cnt
            total_present += len(y)
            
            if temp_cnt != 0:
                l2, l_inf, low_fre, ssim, CIEDE2000, sr, lr = PerD.cal_perceptual_distances(x[success_index], x_adv[success_index])
                PerD.update(l2, l_inf, low_fre, ssim, CIEDE2000, sr, lr, temp_cnt)
            
            pbar.set_postfix_str(f"ASR: {success_cnt / total_present : .5f}, L2: {PerD.l2_avg:.5f}, " + 
                                 f"Linf: {PerD.l_inf_avg : .5f}, SSIM: {PerD.ssim_avg : .5f}, " + 
                                 f"LF:{PerD.LF_avg : .5f}, sr_l2:{PerD.sr_avg : .5f}, " +
                                 f"tr_l2:{PerD.tr_avg : .5f}, ciede2000: {PerD.CIEDE2000_avg : .5f}")
            

if __name__ == "__main__":
    main()
