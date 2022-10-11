""" TokenMixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
import torch
import torchvision



def tokenmix_mask_and_lam(num_patches, lam, correct_lam=True):
    """ Generate bbox and apply lambda correction.
    """
    num_mask = int((1-lam) * num_patches)
    mask = np.hstack([
        np.zeros(num_patches - num_mask),
        np.ones(num_mask),
    ])

    np.random.shuffle(mask)
    if correct_lam:
        lam = (num_patches - num_mask)/ num_patches
    return mask, lam, num_mask

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)

    return y1 * lam + y2 * (1. - lam)

def token_label(target, num_classes, mask, smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y1 = y1.unsqueeze(1).repeat(1,len(mask),1)
    y1[:, mask==1, :] = y1.flip(0)[:, mask==1, :]
    return y1




def rand_token_bbox(img_shape, lam, margin=0., count=None):
    """ Modified CutMix bounding-box(token-level)
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yu = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xu = np.clip(cx + cut_w // 2, 0, img_w)
    bbox_area = (yu - yl) * (xu - xl)
    lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam






class TL_AlignMix:
    """ Modified Cutmix for applying TL-Align

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        tokenmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, mixup_alpha=1., tokenmix_alpha=0., patch_size=16, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.tokenmix_alpha = tokenmix_alpha
        self.patch_size = patch_size
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)



    def _params_per_batch(self):
        lam = 1.
        use_tokenmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.tokenmix_alpha > 0.:
                use_tokenmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.tokenmix_alpha, self.tokenmix_alpha) if use_tokenmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.tokenmix_alpha > 0.:
                use_tokenmix = True
                lam_mix = np.random.beta(self.tokenmix_alpha, self.tokenmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., tokenmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_tokenmix

    def _mix_batch(self, x):

        lam, use_tokenmix = self._params_per_batch()
        while lam == 1.:
            lam, use_tokenmix = self._params_per_batch()

        mask = None
        if use_tokenmix:

            B, C, H, W = x.shape
            x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
            x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # .view(B, C, -1, self.patch_size, self.patch_size)
            (yl, yh, xl, xh), lam = rand_token_bbox(x.shape[2:4], lam)
            x[:, :, yl:yh, xl:xh, :, :] = x.flip(0)[:, :, yl:yh, xl:xh, :, :]
            x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
            mask = torch.zeros((H // self.patch_size, W // self.patch_size))
            mask[yl:yh, xl:xh] = 1
            mask = mask.view(-1)


        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x = x.mul_(lam).add_(x_flipped)
        return lam, x, mask

    def __call__(self, x, target):

        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam, x, mask = self._mix_batch(x)

            cls_target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)


            token_target = token_label(target, self.num_classes, mask, self.label_smoothing, x.device)


            target = torch.cat([cls_target.unsqueeze(1), token_target],dim =1).float()

            return x, target