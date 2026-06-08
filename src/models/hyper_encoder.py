"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn

import manifolds
from src.models.layers import GraphAttentionLayer 
from src.models.layers import GraphConvolution, Linear, get_dim_act
import src.models.hyper_layers as hyp_layers
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds.stereographic import PoincareBall
import pdb

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, cfg):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, cfg.model.manifold)()
        assert cfg.model.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(cfg)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, cfg.train.dropout, act, cfg.model.bias, cfg.model.use_att, cfg.model.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        # mapping into the hyperboloid space
        # pdb.set_trace()
        # if we pass to hyperboloid, we are supposed to add one dimension
        # another problem: why we can directly map this feature into hyperbolic space?
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[..., 0:1], x], dim=-1)
            # print(f"x shape is {x.shape}")
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)



class PoincareGCN(nn.Module):
    def __init__(self, k, cfg):
        super().__init__()
        self.cfg = cfg
        self.manifold = PoincareBall(c=float(k))
        assert cfg.model.num_layers > 1
        pgcn_layers = []
        in_features, out_features, in_edge_feature = get_lgcn_dims(cfg)
        for i in range(cfg.model.num_layers):
            pgcn_layers.append(
                hyp_layers.PoincareGraphNeuralNetwork(self.manifold, in_features[i], out_features[i], in_edge_feature[i], use_layernorm=cfg.model.use_layernorm)
            )

        self.node_linear = hyp_layers.PoincareLinear(self.manifold, in_features[0], out_features[0], bias = True)
        self.edge_linear = hyp_layers.PoincareLinear(self.manifold, in_edge_feature[0], out_features[0], bias = True)
        
        self.layers = nn.Sequential(*pgcn_layers)
        self.encode_graph = True

    def encode(self, x, adj, e, x_manifold='hyp'):
        # todo: 映射到双曲空间的方法是否可以改进
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(x)
            e = self.manifold.expmap0(e)
        
        h_n = self.node_linear.forward(x)
        # print("h_n's nan: ", torch.isnan(h_n).any())
        h_e = self.edge_linear.forward(e)
        # print("h_e's nan: ", torch.isnan(h_e).any())
        input = (h_n, adj, h_e)
        output, _, _ = self.layers.forward(input)
        #print("output's nan: ", torch.isnan(output).any())
        return output
            
            


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
