# Copyright (c) Xi'an Jiaotong University, School of Mechanical Engineering.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae
# SETR: https://github.com/gupta-abhay/setr-pytorch
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class Ssfseg_net(timm.models.vision_transformer.VisionTransformer):
    """
        Modified the head layer
        Added multi head decoding
        decode_layers: Number of multi head decoding layers
    """
    def __init__(self, decode_layers, **kwargs):
        super(Ssfseg_net, self).__init__(**kwargs)

        self.img_size =  kwargs['img_size']
        self.patch_size = kwargs['patch_size']

        self.num_heads = kwargs['num_heads']
        self.head = self.get_classifier()
        self.decode_layers = decode_layers

        self.dec_conv = self.get_dec_conv()

    def forward_features(self, x):
        B = x.shape[0]

        with torch.no_grad(): # # The parameters of self-supervised pre-training are not optimized
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = x[:, 1:, :]

        return x

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_size / self.patch_size),
            int(self.img_size / self.patch_size),
            self.embed_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def get_dec_conv(self):
        modules = []
        for i in range(self.decode_layers):
            modules.append(nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=3,
                stride=1,
                padding=self._get_padding('SAME', (3, 3), ),
                groups=self.num_heads
            ))
            modules.append(nn.BatchNorm2d(self.embed_dim))

        return nn.Sequential(*modules)

    def get_classifier(self):
        extra_in_channels = int(self.embed_dim / self.num_heads)
        modules = []
        in_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            self.num_classes,
        ]

        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=self._get_padding('VALID', (1, 1),),
                )
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        return nn.Sequential(*modules)

    def decode(self, x):
        x = self._reshape_output(x)
        x = self.dec_conv(x)
        x = x.view( x.size(0),
            self.num_heads,
            -1,
            int(self.img_size / self.patch_size),
            int(self.img_size / self.patch_size)).contiguous()
        x = x.sum(dim=1)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.decode(x)
        return x


def Ssfseg_large_patch16(**kwargs):
    model = Ssfseg_net(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == "__main__":
    model = Ssfseg_large_patch16(
            num_classes=9,
            img_size=512,
            decode_layers=2
            )

    finetune = r'./pretarined_weights/objmae_0.75_0.45_512.pth'
    pretrained_weights = torch.load(finetune, map_location='cpu')

    msg = model.load_state_dict(pretrained_weights, strict=False)
    a = torch.rand(1,3,512,512)
    b = model(a)
    print(msg)
    print(b.size())