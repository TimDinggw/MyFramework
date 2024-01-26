import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *

# --- torch_geometric Packages ---
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add
# --- torch_geometric Packages end ---

class Decoder(torch.nn.Module):
    def __init__(self, name, hiddens, skip_conn, train_dist, sampling, alpha, margin):
        super(Decoder, self).__init__()
        print("decoder init")

        self.name = name
        p = 1 if train_dist == "manhattan" else 2
        
        # scoring funciton
        if self.name == "align":
            self.func = Align(p)
        else:
            raise NotImplementedError("bad decoder name: " + self.name)

        if sampling == "T":
            # self.sampling_method = multi_typed_sampling
            self.sampling_method = typed_sampling
        elif sampling == "N":
            self.sampling_method = nearest_neighbor_sampling
        elif sampling == "R":
            self.sampling_method = random_sampling
        elif sampling == ".":
            self.sampling_method = None
        # elif sampling == "SLEF-DESIGN":
        #     self.sampling_method = SLEF-DESIGN_sampling
        else:
            raise NotImplementedError("bad sampling method: " + self.sampling_method)

        # loss function
        if hasattr(self.func, "loss"):
            self.loss = self.func.loss
        else:
            self.loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, ent_emb, rel_emb, sample):
        print("decoder forward")
        if self.name == "align":
            return self.func(ent_emb[sample[:, 0]], ent_emb[sample[:, 1]])
        else:
            raise NotImplementedError("bad decoder name: " + self.name)


# --- Decoding Modules ---
class Align(torch.nn.Module):
    def __init__(self, p):
        super(Align, self).__init__()
        self.p = p

    def forward(self, e1, e2):
        pred = - torch.norm(e1 - e2, p=self.p, dim=1)
        return pred

    def only_pos_loss(self, e1, r, e2):
        return - (F.logsigmoid(- torch.sum(torch.pow(e1 + r - e2, 2), 1))).sum()