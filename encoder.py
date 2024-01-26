import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- torch_geometric Packages ---
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add
# --- torch_geometric Packages end ---

from utils import *

class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, skip_conn, activation, feat_drop, bias, ent_num, rel_num):
        super(Encoder, self).__init__()
        print("encoder init")
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = None
        if activation == 'F.elu':
            self.activation = F.elu
        self.feat_drop = feat_drop
        self.bias = bias
        self.skip_conn = skip_conn
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.rgcn_num_bases = 4

        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=True, bias=bias)
                )
            elif self.name == "mlp":
                self.gnn_layers.append(
                    nn.Linear(self.hiddens[l], self.hiddens[l+1])
                )
            elif self.name == "rgcn":
                self.gnn_layers.append(
                    RGCNConv(self.hiddens[l], self.hiddens[l+1], self.rel_num, num_bases=self.rgcn_num_bases)
                )
            elif self.name == "compgcn":
                self.gnn_layers.append(
                    CompGCNConv(self.hiddens[l], self.hiddens[l+1], self.rel_num, self.feat_drop, self.bias)
                )
            elif self.name == "kecg":
                f_in = self.hiddens[l] * self.heads[l - 1] if l else self.hiddens[l]
                self.gnn_layers.append(
                    #KECGGATConv(self.hiddens[l]*self.heads[l-1], self.hiddens[l+1], self.heads[l], concat=False, bias=self.bias)
                    KECGMultiHeadGraphAttention(self.heads[l], f_in, self.hiddens[l+1], attn_dropout=0.0, init=nn.init.ones_)
                )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
            
        if self.skip_conn == "highway":
            self.gate_weights = nn.ParameterList()
            self.gate_biases = nn.ParameterList()
            for l in range(self.num_layers):
                self.gate_weights.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hiddens[l], self.hiddens[l]))))    
                self.gate_biases.append(nn.Parameter(torch.zeros(self.hiddens[l])))
                
    def forward(self, edges, rels, x, r):
        print("encoder forward")
        edges = edges.t()

        all_layer_outputs = [x]

        if self.skip_conn == "highway":
            highway_features = x

        for l in range(self.num_layers):
            residual = x
            x = F.dropout(x, p=self.feat_drop, training=self.training)

            if self.name == "gcn-align":
                x = self.gnn_layers[l](x, edges)
            elif self.name == "mlp":
                x = self.gnn_layers[l](x)
            elif self.name == "rgcn":
                x = self.gnn_layers[l](x, edges, rels) 
            elif self.name == "compgcn":
                x, r = self.gnn_layers[l](x, edges, rels, r)
            elif self.name == "kecg":
                self.diag = True
                x = self.gnn_layers[l](x, edges)
                if self.diag:
                    x = x.mean(dim=0)

            if self.skip_conn == "residual":
                x += residual

            if l != self.num_layers - 1:
                if self.activation:
                    x = self.activation(x)

            if self.skip_conn == "highway":
                print("highway")
                gate = torch.matmul(highway_features, self.gate_weights[l])
                gate = gate + self.gate_biases[l]
                gate = torch.sigmoid(gate)
                x = x * gate + highway_features * (1.0 - gate)
                
            all_layer_outputs.append(x)

        if self.skip_conn == "concatall":
            print("concatall")
            return torch.cat(all_layer_outputs, dim=1)
        elif self.skip_conn == 'concat0andl':
            print("concat0andl")
            return torch.cat([all_layer_outputs[0], all_layer_outputs[self.num_layers]], dim=1)
        return x   

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))
# --- Main Models: Encoder end ---


# --- Encoding Modules ---
class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
    

class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, feat_drop, bias):
        super(self.__class__, self).__init__()

        self.in_channels	= in_channels
        self.out_channels	= out_channels
        self.num_rels 		= num_rels
        self.device		= None

        self.w_loop		= get_param((in_channels, out_channels))
        self.w_in		= get_param((in_channels, out_channels))
        self.w_out		= get_param((in_channels, out_channels))
        self.w_rel 		= get_param((in_channels, out_channels))
        self.loop_rel 		= get_param((1, in_channels))

        self.drop		= torch.nn.Dropout(feat_drop)
        self.bn			= torch.nn.BatchNorm1d(out_channels)
        self.need_bias = bias
        if self.need_bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        self.opn = "corr"#Composition Operation to be used in CompGCN

    def forward(self, x, edge_index, edge_type, rel_embed): 
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent   = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)

        in_res		= self.propagate(aggr='add', edge_index=self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
        loop_res	= self.propagate(aggr='add', edge_index=self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
        out_res		= self.propagate(aggr='add', edge_index=self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.need_bias: out = out + self.bias
        out = self.bn(out)

        return out, torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if   self.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	= torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)    

# EAkit 中的 KECG ,要求 torch_geometric 版本为 1.5
class KECGGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(KECGGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(1, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.xavier_normal_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = torch.mul(x.repeat((1, self.heads)), self.weight)

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        print(size_i)
        #alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = softmax(alpha, edge_index_i, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
# 原 KECG
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class KECGMultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(KECGMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, edge):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1) # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze())) # edge_e: 1 x E
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda()  if next(self.parameters()).is_cuda else torch.ones(size=(N, 1))) # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)
            output.append(h_prime.unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
