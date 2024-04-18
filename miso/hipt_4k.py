import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.multiprocessing
from torchvision import transforms
torch.multiprocessing.set_sharing_strategy('file_system')
from einops import rearrange, reduce, repeat
from . hipt_model_utils import get_vit256, get_vit4k, tensorbatch2im, eval_transforms


class HIPT_4K(torch.nn.Module):
    """
    HIPT Model (ViT-4K) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
    """
    def __init__(self, 
        model256_path=None,
        model4k_path=None, 
        device256=torch.device('cuda:0'), 
                device4k=torch.device('cuda:0')):

        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
        self.model4k = get_vit4k(pretrained_weights=model4k_path).to(device4k)
        self.device256 = device256
        self.device4k = device4k

    def forward(self, x):
        return self.forward_all(x)[0]
    
    def forward_all(self, x):
        """
        Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT-4K.
        1. x is center-cropped such that the W / H is divisible by the patch token size in ViT-4K (e.g. - 256 x 256).
        2. x then gets unfolded into a "batch" of [256 x 256] images.
        3. A pretrained ViT-256 model extracts the CLS token from each [256 x 256] image in the batch.
        4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
        5. This feature grid is then used as the input to ViT-4K, outputting [CLS]_4K.
        
        Args:
            - x (torch.Tensor): [1 x C x W' x H'] image tensor.
        
        Return:
            - features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
        """
        features_cls256, features_sub256 = self.forward_all256(x)
        features_cls4k, features_sub4k = self.forward_all4k(features_cls256)

        return features_cls4k, features_sub4k, features_sub256

    def forward_all256(self, x):
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H] 
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)
                                          
        features_cls256 = []
        features_sub256 = []
        for mini_bs in range(0, batch_256.shape[0], 256):                       # 3. B may be too large for ViT-256. We further take minibatches of 256.
            minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
            fea_all256 = self.model256.forward_all(minibatch_256).cpu()
            fea_cls256 = fea_all256[:, 0]
            fea_sub256 = fea_all256[:, 1:]
            features_cls256.append(fea_cls256) # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.
            features_sub256.append(fea_sub256)

        features_cls256 = torch.vstack(features_cls256)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
        features_sub256 = torch.vstack(features_sub256)
        features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)  # [1 x 384 x w_256 x h_256]
        features_sub256 = features_sub256.reshape(w_256, h_256, 16, 16, 384).permute(4, 0, 1, 2, 3).unsqueeze(dim=0)  # [1 x 384 x w_256 x h_256 x 16 x 16]
        return features_cls256, features_sub256
    
    def forward_all4k(self, features_cls256):
        __, __, w_256, h_256 = features_cls256.shape
        features_cls256 = features_cls256.to(self.device4k, non_blocking=True)
        features_all4k = self.model4k.forward_all(features_cls256)
        # attn_all4k = self.model4k.get_last_selfattention(features_cls256)
        features_cls4k = features_all4k[:, 0]  # 5. [1 x 192], where 192 == dim of ViT-4K [ClS] token.
        features_sub4k = features_all4k[:, 1:]
        features_sub4k = features_sub4k.reshape(1, w_256, h_256, 192).permute(0, 3, 1, 2)
        return features_cls4k, features_sub4k
    
    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        """
        Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
        are divisible by 256.
        
        (Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
        Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
        
        Args:
            - img (torch.Tensor): [1 x C x W' x H'] image tensor.
            - patch_size (int): Desired patch size to evenly subdivide the image.
        
        Return:
            - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
            - w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
            - h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
        """
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)
        return img_new, w_256, h_256
