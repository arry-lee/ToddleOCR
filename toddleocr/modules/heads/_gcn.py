"""
This code is refer from:
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/gcn.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.create_parameter(
            [in_dim * 2, out_dim], default_initializer=nn.init.XavierUniform()
        )
        self.bias = self.create_parameter(
            [out_dim], is_bias=True, default_initializer=nn.init.Assign([0] * out_dim)
        )

        self.aggregator = MeanAggregator()

    def forward(self, features, A):
        b, n, d = features.shape
        assert d == self.in_dim
        agg_feats = self.aggregator(features, A)
        cat_feats = torch.concat([features, agg_feats], dim=2)
        out = torch.einsum("bnd,df->bnf", cat_feats, self.weight)
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Module):
    def __init__(self, feat_len):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(feat_len, affine=False)
        self.conv1 = GraphConv(feat_len, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(32), nn.Linear(32, 2)
        )

    def forward(self, x, A, knn_inds):
        num_local_graphs, num_max_nodes, feat_len = x.shape

        x = x.reshape([-1, feat_len])
        x = self.bn0(x)
        x = x.reshape([num_local_graphs, num_max_nodes, feat_len])

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k = knn_inds.shape[-1]
        mid_feat_len = x.shape[-1]
        edge_feat = torch.zeros([num_local_graphs, k, mid_feat_len])
        for graph_ind in range(num_local_graphs):
            edge_feat[graph_ind, :, :] = x[graph_ind][torch.Tensor(knn_inds[graph_ind])]
        edge_feat = edge_feat.reshape([-1, mid_feat_len])
        pred = self.classifier(edge_feat)

        return pred
