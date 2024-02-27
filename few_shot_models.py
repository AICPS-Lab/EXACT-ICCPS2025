from collections import OrderedDict

import torch
from torch import nn
from torch.functional import F

from utilities import printc 
class time_FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, model, in_channels=6, pretrained_path=None, cfg=None, device='cpu'):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': True}
        

        # Encoder
        self.encoder = model.to(device)
        self.device = device

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 6 x 100], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x 100], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x 100], list of lists of tensors
            qry_imgs: query images
                N x [B x 6 x 100], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs[0])
        batch_size = supp_imgs[0][0].shape[0]
        mts_size = supp_imgs[0][0].shape[-2]
        assert type(supp_imgs) in (list, torch.Tensor) and type(supp_imgs[0]) in (list, torch.Tensor) and type(supp_imgs[0][0]) == torch.Tensor, \
            "The support images should be a list of lists of tensors, but got supp_imgs: {}, supp_imgs[0]: {}, supp_imgs[0][0]: {}".format(
                type(supp_imgs), type(supp_imgs[0]), type(supp_imgs[0][0]))
        ###### Extract features ######
        imgs_concat = torch.cat([supp_imgs.reshape(-1, *supp_imgs.shape[-2:]), 
                                 qry_imgs.reshape(-1, *qry_imgs.shape[-2:])], dim=0)
        temp = self.encoder.get_embedding(imgs_concat.float().to(self.device))
        img_fts = temp[0]
        fts_size = img_fts.shape[-1:] # this is the length, aka 100 (Time series length)
        
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_ways* n_queries, batch_size, -1, *fts_size)
        # fore_mask = fore_mask.reshape(-1, *fore_mask.shape[-1:])
        # back_mask = back_mask.reshape(-1, *back_mask.shape[-1:])
        

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
            
                            for shot in range(n_shots)] for way in range(n_ways)] # 1 x C (channel embedding)
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)] # 1 x C (channel embedding)
            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts) 
            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=mts_size, mode='linear'))
            ###### Prototype alignment loss ######
            if self.config['align']:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size
        
    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None], dim=1) * scaler
        return dist
        
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H'
            mask: binary mask, expect shape: 1 x H
            
        originally: 
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        # IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2)) + 1e-5)
        return masked_fts
        """
        # interpolate on three dim data:
        fts = F.interpolate(fts, size=mask.shape[-1:], mode='linear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2)) \
            / (mask[None, ...].sum(dim=(2)) + 1e-5)
        return masked_fts
        
    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype
    
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H
            back_mask: background masks for support images
                expect shape: way x shot x H
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3)) + 1e-5)
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-1],
                                          mode='linear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], -1,
                                             device=img_fts.device).long() # not fore or back?
                supp_label[fore_mask[way, shot] >= 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=-1) / n_shots / n_ways
        return loss