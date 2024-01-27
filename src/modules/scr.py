import torch

import torch.nn as nn
import src.modules.scr_modules as SCRModules

from info_nce import InfoNCE
import kornia.augmentation as K

class SCR(nn.Module):

    def __init__(self, 
                 temperature, 
                 mode='training',
                 image_size=96):
        super().__init__()
        style_vgg = SCRModules.vgg
        style_vgg = nn.Sequential(*list(style_vgg.children()))
        self.StyleFeatExtractor = SCRModules.StyleExtractor(
            encoder=style_vgg)  
        self.StyleFeatProjector = SCRModules.Projector()

        if mode == 'training':
            self.StyleFeatExtractor.requires_grad_(True)
            self.StyleFeatProjector.requires_grad_(True)
        else:
            self.StyleFeatExtractor.requires_grad_(False)
            self.StyleFeatProjector.requires_grad_(False)
        
        # NCE Loss
        self.nce_loss = InfoNCE(
            temperature=temperature,
            negative_mode='paired',
        )

        # Pos Image random resize and crop
        self.patch_sampler = K.RandomResizedCrop(
            (image_size, image_size),
            scale=(0.8,1.0),
            ratio=(0.75,1.33))
    
    def forward(self, sample_imgs, pos_imgs, neg_imgs, nce_layers='0,1,2,3,4,5'):
        
        # Get generated image style embedding
        sample_style_embeddings = self.StyleFeatProjector(
            self.StyleFeatExtractor(
                sample_imgs, 
                nce_layers), 
            nce_layers) # out: N * C(2048)

        # Random resize and crop for positive images
        pos_imgs = self.patch_sampler(pos_imgs)
        # Get positive image style embedding
        pos_style_embeddings = self.StyleFeatProjector(
            self.StyleFeatExtractor(
                pos_imgs,
                nce_layers),
            nce_layers)

        # Get negative image style embedding
        _, num_neg, _, _, _ = neg_imgs.shape
        for i in range(num_neg):
            neg_imgs_once = neg_imgs[:, i, :, :]
            neg_style_embeddings_once = self.StyleFeatProjector(
                self.StyleFeatExtractor(
                    neg_imgs_once,
                    nce_layers),
                nce_layers)
            for j, layer_out in enumerate(neg_style_embeddings_once):
                if j == 0:
                    neg_style_embeddings_mid = layer_out[None, :, :]
                else:
                    neg_style_embeddings_mid = torch.cat(
                        [neg_style_embeddings_mid, layer_out[None, :, :]],
                        dim=0)
            if i == 0:
                neg_style_embeddings = neg_style_embeddings_mid[:, :, None, :]
            else:
                neg_style_embeddings = torch.cat(
                    [neg_style_embeddings, neg_style_embeddings_mid[:, :, None, :]],
                    dim=2)
        
        return sample_style_embeddings, pos_style_embeddings, neg_style_embeddings
    
    def calculate_nce_loss(self, sample_s, pos_s, neg_s):
        
        num_layer = neg_s.shape[0]
        neg_s_list = []
        for i in range(num_layer):
            neg_s_list.append(neg_s[i])

        total_scm_loss = 0.
        for layer, (sample, pos, neg) in enumerate(zip(sample_s, pos_s, neg_s_list)):
            scm_loss = self.nce_loss(sample, pos, neg)
            total_scm_loss += scm_loss
        
        return total_scm_loss / num_layer
