# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()

        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()



class SimSam(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, proj_hid=128, proj_out=128, pred_hid=128, pred_out=128):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SimSam, self).__init__()


        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=proj_hid)
        backbone_in_channels = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = torch.nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(backbone_in_channels, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_out),
            nn.BatchNorm1d(proj_out)
        )


        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(),
            nn.Linear(pred_hid, pred_out),
        )

        self.d = D()


    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        #print(im_q.size(), im_k.size())

        out1 = self.encoder_q(im_q).squeeze()
        z1 = self.projection(out1)
        p1 = self.prediction(z1)

        out2 = self.encoder_q(im_k).squeeze()
        z2 = self.projection(out2)
        p2 = self.prediction(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.

        return d1, d2
