from utils.DWT import *
from torchattacks.attack import Attack
from torch import nn
import numpy as np

class LTCA(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttack", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            ll_grad, lh_grad, hl_grad, hh_grad = ll.grad, lh.grad, hl.grad, hh.grad
            ll_hat = ll + ll_grad.sign() * ltcm * self.alpha
            lh_hat = lh + lh_grad.sign() * ltcm * self.alpha
            hl_hat = hl + hl_grad.sign() * ltcm * self.alpha
            hh_hat = hh + hh_grad.sign() * ltcm * self.alpha
            
            adv_images = self.IDWT(ll_hat, lh_hat, hl_hat, hh_hat).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images

class LTCA_LL(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttackLL", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            ll_grad = ll.grad
            ll_hat = ll + ll_grad.sign() * ltcm * self.alpha
            
            adv_images = self.IDWT(ll_hat, lh, hl, hh).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images

class LTCA_LH(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttackLH", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            lh_grad = lh.grad

            lh_hat = lh + lh_grad.sign() * ltcm * self.alpha

            adv_images = self.IDWT(ll, lh_hat, hl, hh).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images

class LTCA_HL(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttackHL", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            hl_grad = hl.grad

            hl_hat = hl + hl_grad.sign() * ltcm * self.alpha

            
            adv_images = self.IDWT(ll, lh, hl_hat, hh).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images

class LTCA_HH(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttackHH", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            hh_grad = hh.grad

            hh_hat = hl + hh_grad.sign() * ltcm * self.alpha

            
            adv_images = self.IDWT(ll, lh, hl, hh_hat).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images


class LTCA_AH(Attack):
    def __init__(self, model, image_size, device, steps=150, alpha=0.05, epsilon=0.01, targeted = False):
        super().__init__("LTCAttackAllH", model)
        self.device = device
        self.steps = steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.targeted = targeted
        self.complexity_layer = nn.Conv2d(3, 3, kernel_size=(3, 3),
                                          stride=1, padding=1, bias=False,
                                          padding_mode="replicate", groups=3)
        w = np.repeat(np.asarray([[0.125, 0.125, 0.125]], dtype=np.float32), 3, axis=0).reshape(1, 1, 3, 3)
        w[0, 0, 1, 1] = -1.0
        w = np.repeat(w, 3, axis=0)
        self.complexity_layer.weight = torch.nn.Parameter(torch.from_numpy(w))
        self.pooling_layer = nn.AdaptiveMaxPool2d(int(image_size / 2))
        self.DWT = DWT_2D("haar")
        self.IDWT = IDWT_2D("haar")


    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            targeted_labels = self.get_random_target_label(images, labels)
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            self.complexity_layer = self.complexity_layer.to(device=self.device)
            ltcm = self.complexity_layer(images).abs().detach()
            ltcm = self.pooling_layer(ltcm).detach()
        
        for i in range(0, self.steps):
            ll, lh, hl, hh = self.DWT(adv_images)
            for coefficient in [ll, lh, hl, hh]:
                coefficient.requires_grad_()
            x_reconstruct = self.IDWT(ll, lh, hl, hh)
            output = self.model(x_reconstruct)
            
            if self.targeted:
                loss = -loss_fn(output, targeted_labels)
            else:
                loss = loss_fn(output, labels)
            # print(f"iter {i} : loss: {loss.item()}")
            # ll_grad, lh_grad, hl_grad, hh_grad = torch.autograd.grad(loss, [ll, lh, hl, hh], retain_graph=False, create_graph=False)
            loss.backward()
            lh_grad, hl_grad, hh_grad = lh.grad, hl.grad, hh.grad
            
            lh_hat = lh + lh_grad.sign() * ltcm * self.alpha
            hl_hat = hl + hl_grad.sign() * ltcm * self.alpha
            hh_hat = hh + hh_grad.sign() * ltcm * self.alpha
            
            adv_images = self.IDWT(ll, lh_hat, hl_hat, hh_hat).detach()

            
            adv_images = self.IDWT(ll, lh, hl_hat, hh).detach()
            
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            with torch.no_grad():
                pred = self.model(adv_images).argmax(dim=1)
            
            if self.targeted:
                fooling_rate = sum(pred == targeted_labels) / 1.0 / len(labels)
            else:
                fooling_rate = sum(pred != labels) / 1.0 / len(labels)
            
            if fooling_rate == 1.0:
                return adv_images

        return adv_images