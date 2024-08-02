import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class MCAB(nn.Module):
    def __init__(self,dim=512):
        super(MCAB, self).__init__()
        self.dconv1 = nn.Conv2d(dim,dim,kernel_size=3,stride=1, padding=3 // 2, groups=dim)
        self.pconv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        self.dconv2 = nn.Conv2d(dim,dim,kernel_size=5,stride=1, padding=5 // 2, groups=dim)
        self.pconv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        self.eca_layer = ECA_Layer(dim)

    def forward(self,x):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]

        _H, _W = int(np.ceil(np.sqrt(N-1))), int(np.ceil(np.sqrt(N-1)))
        add_length = _H * _W - (N - 1)
        zero_tensor = torch.zeros(B, add_length, C).cuda()
        x_pad = torch.cat([feat_token, zero_tensor], dim=1) # [B,_H*_W,C]
        cnn_feat = x_pad.transpose(1, 2).view(B, C, _H, _W)
        x3_3 = self.pconv1(self.dconv1(cnn_feat))
        x5_5 = self.pconv2(self.dconv2(cnn_feat))
        x = cnn_feat + x3_3 + x5_5
        y = self.eca_layer(x)
        x = y * x3_3 + y * x5_5
        x = x.flatten(2).transpose(1, 2)    # [B,_H*_W,C]
        x = x[:, :N-1]
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class ECA_Layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)


        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        y = self.sigmoid(y)

        return y

class CAMIL(nn.Module):
    def __init__(self, n_classes):
        super(CAMIL, self).__init__()
        self.pos_layer = MCAB(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):
        h = kwargs['data'].float()

        h = self._fc1(h)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)


        h = self.layer1(h)

        h = h + self.pos_layer(h)


        h = self.layer2(h)


        h = self.norm(h)[:, 0]


        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

