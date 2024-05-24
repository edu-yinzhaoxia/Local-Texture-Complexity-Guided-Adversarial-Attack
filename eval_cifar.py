import torch
from torchvision.datasets import CIFAR10
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.auxiliary_utils import *
from utils.eval_metric_utils import *
from LTCA import LTCA
from models import ResNet18
import torch.backends.cudnn as cudnn


def main():
    alpha = 0.05
    epsilon = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    random_seed = 1

    data_dir = "../datasets/CIFAR10"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_data = CIFAR10(root=data_dir, 
                          train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    resnet_model = ResNet18()
    resnet_model = resnet_model.to(device)
    if device == 'cuda':
        resnet_model = torch.nn.DataParallel(resnet_model, device_ids=[0])
        cudnn.benchmark = True
    checkpoint = torch.load("./ckpts/resnet18.pth")
    resnet_model.load_state_dict(checkpoint['net'])
    resnet_model = nn.Sequential(
        norm_layer,
        resnet_model
    ).to(device)
    resnet_model.eval()

    attack = LTCA(model=resnet_model, image_size=32, device=device, alpha=alpha, epsilon=epsilon)
    PerD = PerceptualDistance("haar", image_size=32)
    
    torch.manual_seed(random_seed)
    
    success_cnt, total_present = 0, 0
    
    with tqdm(test_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = resnet_model(x).argmax(dim=1)
                if pred != y:
                    # print(f"pred is : {pred.item()}, true is : {y.item()}")
                    continue
            x_adv = attack(x, y)
            
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
                                 f"ciede2000: {PerD.CIEDE2000_avg : .5f}, LF:{PerD.LF_avg : .5f}, " +
                                 f"sr_l2:{PerD.sr_avg : .5f}, tr_l2:{PerD.tr_avg : .5f}")


if __name__ == "__main__":
    main()
