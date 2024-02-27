import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from utilities import printc


class Segmenter(nn.Module):
    def __init__(self, in_channels=6, num_layers=2, num_heads=4, embed_dims=256, num_classes=7, **kwargs):
        super(Segmenter, self).__init__(**kwargs)

        # Fixed parameters for simplicity
        mlp_ratio = 4
        # norm_cfg = dict(type='LN')
        # act_cfg = dict(type='GELU')
        self.init_std = 0.02
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=mlp_ratio * embed_dims,
                dropout=0.5,
                activation=F.gelu,
                layer_norm_eps=1e-05,
                batch_first=True,
                norm_first=False,
                bias=True
            )
            self.layers.append(layer)

        self.dec_proj = nn.Linear(in_channels, embed_dims)
        self.cls_emb = nn.Parameter(torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.att_norm = nn.LayerNorm(embed_dims)
        self.decoder_norm = nn.LayerNorm(embed_dims)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.init_weights()
    def init_weights(self):
        nn.init.trunc_normal_(self.cls_emb, std=self.init_std)
        nn.init.trunc_normal_(self.patch_proj.weight, std=self.init_std)
        nn.init.trunc_normal_(self.classes_proj.weight, std=self.init_std)
        # Initialize weights for Transformer layers
        for layer in self.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1) # b h c
        b, c, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers: 
            # layer norm:
            x = self.att_norm(x)
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes]) # shape 128, 75, 64 (B, Time_length, embedding)
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:]) # shape 128, 7, 64 (B, num_classes, embedding)
        # cls_seg_feat = nn.functional.dropout(cls_seg_feat, p=0.5, training=self.training)
        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks).contiguous().view(b, h, -1)

        return masks
    def get_embedding(self, inputs):
        x = inputs
        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers: 
            # layer norm:
            x = self.att_norm(x)
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes]) # shape 128, 75, 64 (B, Time_length, embedding)
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])
        return patches, cls_seg_feat 

    def forward_pred(self, inputs):
        masks = self.forward(inputs)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
