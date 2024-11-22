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
    def __init__(self,dim=512,temperature=3):
        super().__init__()

        self.local_feature3 = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,stride=1, padding=3 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        )

        self.local_feature5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1)
        )


        ## self.eca_layer = ECA_Layer(dim)
        # self.cam = CAM_1(n_channel=dim,mlp_r=2,temperature=temperature)
        self.cam = CAM(n_channel=dim, mlp_r=2, temperature=temperature)

    def forward(self,x):
        B, N, C = x.shape

        _H, _W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = _H * _W - N
        zero_tensor = torch.zeros(B, add_length, C,device=x.device)
        
        x_pad = torch.cat([x, zero_tensor], dim=1) # [B,_H*_W,C]

        cnn_feat = x_pad.transpose(1, 2).view(B, C, _H, _W)
        x3_3 = self.local_feature3(cnn_feat)
        x5_5 = self.local_feature5(cnn_feat)
        x = cnn_feat + x3_3 + x5_5

        y1,y2,y3 = self.cam(x) # [B,C,1,1]
        x = y1 * cnn_feat + y2 * x3_3 + y3 * x5_5
        x = x.flatten(2).transpose(1, 2)    # [B,_H*_W,C]
        x = x[:, :N]
        return x

class CAM(nn.Module):
    def __init__(self,n_channel,mlp_r=2,temperature=1):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(n_channel, n_channel // mlp_r),
            nn.ReLU(),
            nn.Linear(n_channel // mlp_r, 3 * n_channel)
        )
        self.temperature = temperature

    def forward(self,x):
        B,C,_,_ = x.shape
        max = self.max_pool(x)
        avg = self.avg_pool(x)
        max = self.linear(max.view(B,C)).view(B,3*C,1,1)
        avg = self.linear(avg.view(B,C)).view(B,3*C,1,1)
        y = torch.sigmoid((max + avg) * self.temperature)
        y_1 = y[:,:C]
        y_2 = y[:,C:2*C]
        y_3 = y[:,2*C:]
        return y_1,y_2,y_3

class Block(nn.Module):
    def __init__(self,dim=512,temperature=3):
        super().__init__()
        self.trans_layer = TransLayer(dim=dim)
        self.mcab = MCAB(dim=dim,temperature=temperature)

    def forward(self,x):
        x = self.trans_layer(x)
        x = self.mcab(x)

        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):

    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CAMIL(nn.Module):
    def __init__(self, n_classes, input_dim=1024, temperature=3, dropout=0.25, n_layers=4, gate=True):
        super().__init__()
        # self.pos_layer = MCAB(dim=512,temperature=temperature)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        # self.layer1 = TransLayer(dim=512)
        # self.layer2 = TransLayer(dim=512)
        self.layers = nn.ModuleList([
            Block(dim=512,temperature=temperature) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(512)

        if gate:
            attention_net = Attn_Net_Gated(L=512, D=256, dropout=False, n_classes=1)
        else:
            attention_net = Attn_Net(L=512, D=256, dropout=False, n_classes=1)

        self.attention_net = attention_net
        self._fc2 = nn.Linear(512, self.n_classes)
        initialize_weights(self)

    def forward(self, **kwargs):
        h = kwargs['data'].float()  # [B, n, 1024]
        h = self._fc1(h)  # [B, n, 512]

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        h = h.squeeze(0)
        # ---->abmil pool
        A, h = self.attention_net(h)
        A = A.permute(1,0)
        A = torch.softmax(A,dim=1)
        h = torch.mm(A,h) 
        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

