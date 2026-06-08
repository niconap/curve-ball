import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.models.hyper_layers import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypCLS, PoincareLinear, PoincareLayerNorm, PoincareDropout
from src.models.hyper_encoder import HGCN, LGCN, PoincareGCN
from src.models.hyper_decoder import FermiDiracDecoder
# from src.manifolds.poincareball import PoincareBall
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
from utils import check_on_manifold
from src.models.transormer_layers import TimestepEmbedder
# from geoopt.manifolds.euclidean import Euclidean
from src.manifolds.euclidean import Euclidean2 as Euclidean
from geoopt.manifolds.product import ProductManifold
from src.models.vq import VectorQuantize
from src.models.PoincareTransformer import PoincareTransformer
import wandb
import geoopt

class PoincareMLP(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """

    def __init__(self, manifold, input, layers=[16, 16], dropout_rate=0, use_resnet=False, use_layernorm=False):
        """
        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(PoincareMLP, self).__init__()
        self.manifold = manifold
        self.layers = torch.nn.ModuleList(
            [PoincareLinear(self.manifold, input, layers[0])])
        for i in range(len(layers)-1):
            self.layers.append(PoincareLinear(
                self.manifold, layers[i], layers[i+1]))

        self.norm_layers = None
        if use_layernorm:
            self.norm_layers = torch.nn.ModuleList(
                [PoincareLayerNorm(manifold, c) for c in [input]+layers])
        self.dropout = PoincareDropout(manifold, dropout_rate)
        self.use_resnet = use_resnet
        # self.reset_parameters()

    def forward(self, in_tensor, applyActOnTheLastLyr=False):
        h = in_tensor
        for i in range(len(self.layers)):
            residual = h  # Save the input for the residual connection
            if self.norm_layers != None:
                if len(h.shape) == 3:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h = h.reshape(h.shape[0], -1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h = h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)

            if self.use_resnet and residual.shape == h.shape:
                h = self.manifold.mobius_add(residual, h)

        return h

class EucEncoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        self.node_mlp = nn.Sequential(
            nn.Linear(self.euc_channels, 4*self.euc_channels),
            nn.ReLU(),
            nn.Linear(4*self.euc_channels, self.out_channels)
        )
        self.node_mlp2 = PoincareMLP(self.manifold, input=self.euc_channels, layers=[
                                     self.euc_channels, self.euc_channels], dropout_rate=0.1, use_resnet=False, use_layernorm=False)

    def forward(self, z):
        z = self.node_mlp2(z)
        z = self.manifold.logmap0(z)
        node_feat = self.node_mlp(z)
        return node_feat


class Euc2NodeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU()
        )

        # after mlp, hypmlp will not change the output channel now, the output includes the time dimension to predict the res
        self.linear = nn.Linear(self.euc_channels, self.out_channels)

    def forward(self, z):
        node_feat = self.mlp(z)
        node_feat = self.linear(node_feat)
        return node_feat


class Hyp2NodeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
            nn.LayerNorm(self.euc_channels),
            nn.ReLU()
        )

        # after mlp, hypmlp will not change the output channel now, the output includes the time dimension to predict the res
        self.linear = nn.Linear(self.euc_channels, self.out_channels)
        self.edge_mlp = PoincareMLP(self.manifold, input=self.in_channels, layers=[
                                    self.in_channels, self.in_channels], dropout_rate=0.1, use_resnet=use_resnet, use_layernorm=use_layernorm)

    def forward(self, z):
        z = self.edge_mlp(z)
        z = self.manifold.logmap0(z)
        node_feat = self.mlp(z)
        node_feat = self.linear(node_feat)
        return node_feat


class Euc2EdgeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, method="pairwise_interaction", init_thresholds=None, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        self.decoder = nn.Sequential(
            nn.Linear(self.in_channels, self.euc_channels),
            nn.ReLU(),
            nn.Linear(self.euc_channels, self.euc_channels),
        )

        self.prediction_head3 = nn.Linear(self.euc_channels, self.euc_channels)
        self.prediction_head2 = nn.Linear(self.euc_channels, self.out_channels)

        self.method = method
        if self.method == "pairwise_interaction":
            self.interaction_mlp = nn.Sequential(
                nn.Linear(2 * self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )
            # Edge classification layer
            self.edge_classifier = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "pairwise_distance":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "inner_product":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "fermi_dirac":
            self.r = nn.Parameter(torch.tensor([2.0], requires_grad=True))
            self.t = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.fermi_dirac = FermiDiracDecoder(self.r, self.t)

        elif self.method == "logmap":
            self.prediction_head = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )

    def hyper_distance(self, x, y):
        return self.manifold.cdist(x, y)

    def forward(self, z):
        z = self.decoder(z)
        if self.method == "pairwise_interaction":
            B, N, D1 = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            z_j = z.unsqueeze(1).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            # Shape: (B, N, N, 2 * D1)
            pair_features = torch.cat([z_i, z_j], dim=-1)

            h = self.interaction_mlp(pair_features)
            edge_feat = self.edge_classifier(h)
            return edge_feat

        elif self.method == "fermi_dirac":
            distance = self.hyper_distance(z, z)
            edge_feat = self.fermi_dirac(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat

        elif self.method == "inner_product":
            distance = self.manifold.inner(z, z)
            edge_feat = self.prediction_head(
                distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat

        elif self.method == "pairwise_distance":
            distance = self.hyper_distance(z, z)
            edge_feat = self.prediction_head(
                distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat

        elif self.method == "logmap":
            B, N, D = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]

            logmap_vectors = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]

            edge_feat = self.prediction_head(logmap_vectors)
            return edge_feat


class Hyp2EdgeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, euc_channels, out_channels, method="pairwise_interaction", init_thresholds=None, use_resnet=False, use_layernorm=False):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.euc_channels = euc_channels
        self.out_channels = out_channels
        # self.mlp = LorentzMLP(self.manifold, input= 2 * self.euc_channels, layers=[self.euc_channels, self.out_channels+1], normalize=True, dropout_rate=0)

        if isinstance(self.manifold, Lorentz):
            self.decoder = LorentzMLP(self.manifold, input=self.in_channels, layers=[
                                      self.euc_channels, self.euc_channels], normalize=True, dropout_rate=0)
            self.act = HypActivation(self.manifold, activation=F.relu)
        else:
            self.decoder = PoincareMLP(self.manifold, input=self.in_channels, layers=[
                                       self.euc_channels, self.euc_channels], dropout_rate=0, use_resnet=use_resnet, use_layernorm=use_layernorm)

        self.method = method
        if self.method == "pairwise_interaction":
            self.interaction_mlp = nn.Sequential(
                nn.Linear(2 * self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels)
            )
            # Edge classification layer
            self.edge_classifier = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "pairwise_distance":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "inner_product":
            self.prediction_head = nn.Sequential(
                nn.Linear(1, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "logmap" or self.method == "logmap_symmetry":
            self.prediction_head = nn.Sequential(
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.Linear(self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "fermi_dirac":
            self.r = nn.Parameter(torch.tensor([2.0], requires_grad=True))
            self.t = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.fermi_dirac = FermiDiracDecoder(self.r, self.t)

        elif self.method == "tangent_mlp":
            self.tan_mlp = nn.Sequential(
                nn.Linear(3*self.euc_channels, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )

        elif self.method == "strong_mlp":
            self.strong_mlp = nn.Sequential(
                nn.Linear(3*self.euc_channels + 2, self.euc_channels),
                nn.ReLU(),
                nn.Linear(self.euc_channels, self.out_channels)
            )
        elif self.method == "transh":
            self.R_mat = nn.Parameter(
                torch.eye(self.euc_channels).repeat(self.R, 1, 1))  # 旋转
            self.t_vec = nn.Parameter(torch.zeros(
                self.R, self.euc_channels))          # 位移
        elif self.method == "gyroplane":
            self.n_vec = nn.Parameter(torch.randn(
                self.out_channels, self.euc_channels))
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        elif self.method == "multirel_distance":
            self.rel_emb = geoopt.ManifoldParameter(
                manifold.random_normal((self.R, self.euc_channels)),
                manifold=self.manifold)
            # global scale / bias for 1.1
            self.alpha = nn.Parameter(torch.tensor(10.0))
            self.beta = nn.Parameter(torch.tensor(0.0))

    def hyper_distance(self, x, y):
        return self.manifold.cdist(x, y)

    def forward_multirel_distance(self, z):
        """K-class edge type；learn K relation embeddings r_k"""
        z = self.decoder(z)
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        rel = self.rel_emb.view(1, 1, 1, self.R, D)         # [1,1,1,R,D]
        z_i_r = self.manifold.mobius_add(z_i.unsqueeze(3), rel)  # [B,N,N,R,D]
        dist = self.manifold.dist(z_i_r, z_j.unsqueeze(3))       # [B,N,N,R]
        score = -self.alpha * dist                              # R-way logit
        return score                                            # CrossEntropyLoss

    def forward_gyroplane(self, z):
        """超平面 Logistic / Softmax"""
        z = self.decoder(z)
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        v = self.manifold.logmap0(self.manifold.mobius_add(
            self.manifold.mobius_neg(z_i), z_j))   # [B,N,N,D]

        logits = torch.einsum('b m n d , k d -> b m n k',
                              v, self.n_vec) + self.bias
        return logits                              # binary: k=1;  multi-class: k=R

    def forward_tangent_mlp(self, z):
        """logmap0 → Euclidean MLP"""
        z = self.decoder(z)
        v = self.manifold.logmap0(z)               # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        logmap_vectors1 = self.manifold.logmap(z_i, z_j)
        B, N, D = v.shape
        v_i = v.unsqueeze(2).expand(B, N, N, D)
        v_j = v.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([v_i, v_j, logmap_vectors1], dim=-1)   # [B,N,N,3D]
        out = self.tan_mlp(pair)
        return out

    def forward_strong_mlp(self, z):
        """logmap0 → Euclidean MLP"""
        z = self.decoder(z)
        v = self.manifold.logmap0(z)               # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        logmap_vectors1 = self.manifold.logmap(z_i, z_j)
        # calculate distance
        dist = self.manifold.dist(z_i, z_j).unsqueeze(-1)  # [B,N,N,1]
        B, N, D = v.shape
        v_i = v.unsqueeze(2).expand(B, N, N, D)
        v_j = v.unsqueeze(1).expand(B, N, N, D)
        # calculate angle between vi and vj,
        v_i_unit = F.normalize(v_i, dim=-1, eps=1e-7)
        v_j_unit = F.normalize(v_j, dim=-1, eps=1e-7)
        cos_angle = (v_i_unit * v_j_unit).sum(-1, keepdim=True)\
            .clamp(-1.0 + 1e-6, 1.0 - 1e-6)        # [B,N,N,1]

        pair = torch.cat([v_i, v_j, logmap_vectors1, dist,
                         cos_angle], dim=-1)   # [B,N,N,3D + 2]
        out = self.strong_mlp(pair)
        return out

    def forward_transh(self, z):
        """关系平移 / 旋转后再距"""
        z = self.decoder(z)                        # [B,N,D]
        B, N, D = z.shape
        z_i = z.unsqueeze(2).expand(B, N, N, D)
        z_j = z.unsqueeze(1).expand(B, N, N, D)

        z_i_r = self.trans_rotate(z_i, self.R_mat)     
        z_i_r = self.manifold.mobius_add(z_i_r, self.t_vec)  
        dist = -self.manifold.dist(z_i_r, z_j)         # [B,N,N,R]
        # pdb.set_trace()
        return dist                                    # CrossEntropyLoss

    def forward(self, z):
        if isinstance(self.manifold, Lorentz):
            z = self.decoder(z, activation=self.act)
        else:
            z = self.decoder(z)

        if self.method == "pairwise_interaction":
            z = self.manifold.logmap0(z)
            B, N, D1 = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            z_j = z.unsqueeze(1).expand(B, N, N, D1)  # Shape: (B, N, N, D1)
            # Shape: (B, N, N, 2 * D1)
            pair_features = torch.cat([z_i, z_j], dim=-1)

            h = self.interaction_mlp(pair_features)
            edge_feat = self.edge_classifier(h)
            return edge_feat

        elif self.method == "fermi_dirac":
            distance = self.hyper_distance(z, z)
            edge_feat = self.fermi_dirac(distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat

        elif self.method == "inner_product":
            distance = self.manifold.inner(z, z)
            edge_feat = self.prediction_head(
                distance.unsqueeze(-1)).squeeze(-1)
            return edge_feat

        elif self.method == "pairwise_distance":
            distance = -self.hyper_distance(z, z)
            edge_feat = self.prediction_head(
                distance.unsqueeze(-1)).squeeze(-1)
            edge_feat = torch.sigmoid(edge_feat)
            return edge_feat

        elif self.method == "logmap":
            B, N, D = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]

            logmap_vectors = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]

            edge_feat = self.prediction_head(logmap_vectors)
            return edge_feat

        elif self.method == "logmap_symmetry":
            B, N, D = z.shape
            z_i = z.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
            z_j = z.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]

            logmap_vectors1 = self.manifold.logmap(z_i, z_j)  # [B, N, N, D]
            logmap_vectors2 = self.manifold.logmap(z_j, z_i)  # [B, N, N, D]

            edge_feat = self.prediction_head(
                logmap_vectors1) + self.prediction_head(logmap_vectors2)
            return edge_feat

        elif self.method == 'pairwise_distance_logit':
            logits = self.forward_pairwise_distance_logit(z)

        elif self.method == 'multirel_distance':
            logits = self.forward_multirel_distance(z)      # [B,N,N,R]

        elif self.method == 'gyroplane':
            logits = self.forward_gyroplane(z)

        elif self.method == 'tangent_mlp':
            logits = self.forward_tangent_mlp(z)

        elif self.method == 'transh':
            logits = self.forward_transh(z)

        elif self.method == 'strong_mlp':
            logits = self.forward_strong_mlp(z)

        return logits


class HypFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.use_VAE = (cfg.loss.lambda_kl != 0)
        self.use_VQVAE = (cfg.loss.lambda_commitment_weight != 0)
        self.in_channels = cfg.model.latent_channels
        self.euc_channels = cfg.model.euc_channels
        self.hyp_channels = cfg.model.hyp_channels
        self.latent_channels = cfg.model.latent_channels

        self.manifold_in = Lorentz(k=float(cfg.model.k_in))
        self.manifold_hidden = Lorentz(k=float(cfg.model.k_hidden))
        if cfg.model.use_poincare:
            self.manifold_out = PoincareBall(
                c=1/float(self.cfg.model.k_poin_out))  # poincare ball上曲率暂时不变
        else:
            self.manifold_out = Lorentz(k=float(cfg.model.k_out))

        if self.cfg.model.use_poincare:
            self.manifold = PoincareBall(c=1/float(self.cfg.model.k_poin_out))
        else:
            self.manifold = Lorentz(k=float(cfg.model.k_out))
        # self.product_manifold = ProductManifold((Euclidean(), self.euc_channels), (self.manifold, self.hyp_channels))
        self.product_manifold = self.manifold
        self.extra_feature_mlp = nn.Sequential(
            nn.Linear(self.cfg.model.lgcn_in_channels, self.latent_channels),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels),
            nn.LayerNorm(self.latent_channels)
        )
        self.extra_edge_feature_mlp = nn.Sequential(
            nn.Linear(self.cfg.model.lgcn_in_edge_channels,
                      self.latent_channels),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels),
            nn.LayerNorm(self.latent_channels)
        )

        if cfg.model.use_poincare:
            self.hyper_graph_conv = PoincareGCN(cfg.model.k_poin_out, cfg)
        else:
            self.hyper_graph_conv = LGCN(cfg.model.k_in, cfg)

        if cfg.model.transformer_encoder.Hypformer_use:
            self.trans_conv_linear = PoincareTransformer(cfg, self.manifold, self.in_channels, cfg.model.transformer_encoder.trans_num_layers, cfg.model.transformer_encoder.trans_num_heads, cfg.model.transformer_encoder.trans_dropout,
                                                         cfg.model.transformer_encoder.max_seq_len, cfg.model.transformer_encoder.use_hyperbolic_attention, cfg.model.transformer_encoder.attention_type, cfg.model.transformer_encoder.attention_activation,
                                                         use_pe=cfg.model.transformer_encoder.use_pe)
        else:
            if cfg.model.use_poincare:
                self.trans_conv_linear = PoincareLinear(
                    self.manifold, self.in_channels, self.latent_channels)
                self.layernorm = PoincareLayerNorm(
                    self.manifold_out, self.latent_channels)
            else:
                self.trans_conv_linear = LorentzHypLinear(
                    self.manifold_out, self.in_channels, self.latent_channels)
                self.layernorm = HypLayerNorm(
                    self.manifold_out, self.latent_channels)

        if self.euc_channels == 0 and cfg.model.transformer_decoder.Hypformer_use:
            self.trans_decoder = PoincareTransformer(cfg, self.manifold, self.hyp_channels, cfg.model.transformer_decoder.trans_num_layers, cfg.model.transformer_decoder.trans_num_heads, cfg.model.transformer_decoder.trans_dropout,
                                                     cfg.model.transformer_decoder.max_seq_len, cfg.model.transformer_decoder.use_hyperbolic_attention, cfg.model.transformer_decoder.attention_type, cfg.model.transformer_decoder.attention_activation,
                                                     use_pe=cfg.model.transformer_decoder.use_pe)
        if self.use_VQVAE:
            if self.euc_channels > 0:
                self.euc_encoder = EucEncoder(
                    self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                self.euc_codebook = VectorQuantize(
                    self.euc_channels, self.cfg.model.codebook_size)
            if self.hyp_channels > 0:
                self.hyp_encoder = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels,
                                              self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                self.hyp_codebook = VectorQuantize(self.hyp_channels, self.cfg.model.codebook_size, vq_loss_weight=self.cfg.loss.lambda_vq_loss_weight,
                                                   commitment_weight=self.cfg.loss.lambda_commitment_weight, orthogonal_reg_weight=self.cfg.loss.lambda_orthogonal_reg_weight,
                                                   use_hyperbolic=cfg.model.use_poincare, manifold=self.manifold, kmeans_init=cfg.loss.use_kmeans, sample_codebook_temp=cfg.model.sample_codebook_temp)
        else:
            if self.use_VAE:
                if self.euc_channels > 0:
                    self.euc_encoder_mean = EucEncoder(
                        self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                    self.euc_encoder_var = EucEncoder(
                        self.manifold_out,  self.latent_channels, self.latent_channels, self.euc_channels)
                if self.hyp_channels > 0:
                    self.hyp_encoder_mean = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels,
                                                       self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                    self.hyp_encoder_var = EucEncoder(
                        self.manifold_out, self.latent_channels, self.latent_channels, self.hyp_channels, )
            else:
                if self.euc_channels > 0:
                    self.euc_encoder = EucEncoder(self.manifold_out,  self.latent_channels, self.latent_channels,
                                                  self.euc_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
                if self.hyp_channels > 0:
                    self.hyp_encoder = HypEncoder(self.manifold_out, self.latent_channels, self.latent_channels,
                                                  self.hyp_channels, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)

        self.decode_method = cfg.model.decode_method
        if self.euc_channels > 0:
            self.euc2node_decoder = Euc2NodeDecoder(self.manifold_out,  self.euc_channels, self.euc_channels,
                                                    cfg.model.node_classes, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
            self.euc2edge_decoder = Euc2EdgeDecoder(Euclidean(),  self.euc_channels, self.euc_channels, cfg.model.edge_classes,
                                                    method=self.decode_method, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
        if self.hyp_channels > 0:
            self.hyp2node_decoder = Hyp2NodeDecoder(self.manifold_out,  self.hyp_channels, self.hyp_channels,
                                                    cfg.model.node_classes, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)
            self.hyp2edge_decoder = Hyp2EdgeDecoder(self.manifold_out,  self.hyp_channels, self.hyp_channels, cfg.model.edge_classes,
                                                    method=self.decode_method, use_resnet=cfg.model.use_resnet, use_layernorm=cfg.model.use_layernorm)

    def forward(self, dataset):

        raise NotImplementedError

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def encode(self, x, adj, e, mask):
        # pdb.set_trace()
        x_dim = x.shape[-1]
        x = self.extra_feature_mlp(x)
        e = self.extra_edge_feature_mlp(e)

        x = self.hyper_graph_conv.encode(x, adj, e, x_manifold='euc')
        x_norm = torch.norm(x, p=2, dim=-1)
        epoch = wandb.run.summary.get("epoch", 0)
        to_log = {
            "epoch": epoch,
            "train/after_GNN_x_norm_min": x_norm.min().item(),
            "train/after_GNN_x_norm_max": x_norm.max().item(),
            "train/after_GNN_x_norm_mean": x_norm.mean().item()
        }
        wandb.log(to_log)
        print(
            f"After GNN x_norm norm min: {x_norm.min().item():.4f}, After GNN norm max: {x_norm.max().item():.4f}, After GNN  norm mean: {x_norm.mean().item():.4f}")
        if self.cfg.model.transformer_encoder.Hypformer_use:
            z = self.trans_conv_linear(x, mask)
            z_norm = torch.norm(z, p=2, dim=-1)
            print(
                f"After Transformer x_norm norm min: {z_norm.min().item():.4f}, After Transformer norm max: {z_norm.max().item():.4f}, After Transformer  norm mean: {z_norm.mean().item():.4f}")
            to_log.update({
                "train/after_Transformer_x_norm_min": z_norm.min().item(),
                "train/after_Transformer_x_norm_max": z_norm.max().item(),
                "train/after_Transformer_x_norm_mean": z_norm.mean().item()
            })
        else:
            z = self.trans_conv_linear(x)
        if self.use_VQVAE:
            if self.euc_channels > 0:
                node_feat = self.euc_encoder(z)
            else:
                node_feat = None
            if self.hyp_channels > 0:
                edge_feat = self.hyp_encoder(z)
                # edge_feat = z
                edge_feat_norm = torch.norm(edge_feat, p=2, dim=-1)
                print(
                    f"After HypEncoder edge_feat_norm norm min: {edge_feat_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_norm.mean().item():.4f}")
                to_log.update({
                    "train/after_HypEncoder_edge_feat_norm_min": edge_feat_norm.min().item(),
                    "train/after_HypEncoder_edge_feat_norm_max": edge_feat_norm.max().item(),
                    "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_norm.mean().item()
                })
            else:
                edge_feat = None
            wandb.log(to_log)
            return node_feat, edge_feat, z

        else:
            if not self.cfg.model.transformer_encoder.Hypformer_use:
                # # print z's abs mean
                z = self.layernorm(z)
            if not self.use_VAE:
                if self.euc_channels > 0:
                    node_feat = self.euc_encoder(z)
                else:
                    node_feat = None
                if self.hyp_channels > 0:
                    edge_feat = self.hyp_encoder(z)
                else:
                    edge_feat = None
                edge_feat_norm = torch.norm(edge_feat, p=2, dim=-1)
                print(
                    f"After HypEncoder edge_feat_norm norm min: {edge_feat_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_norm.mean().item():.4f}")
                to_log.update({
                    "train/after_HypEncoder_edge_feat_norm_min": edge_feat_norm.min().item(),
                    "train/after_HypEncoder_edge_feat_norm_max": edge_feat_norm.max().item(),
                    "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_norm.mean().item()
                })
                wandb.log(to_log)
                return node_feat, edge_feat, z

            else:
                if self.euc_channels > 0:
                    node_feat_mean = self.euc_encoder_mean(z)
                    # should layernorm inside the euclidean space
                    node_feat_var = self.euc_encoder_var(z)
                else:
                    node_feat_mean = None
                    node_feat_var = None
                if self.hyp_channels > 0:
                    edge_feat_mean = self.hyp_encoder_mean(z)
                    edge_feat_var = self.hyp_encoder_var(z)
                    edge_feat_mean_norm = torch.norm(
                        edge_feat_mean, p=2, dim=-1)
                    print(
                        f"After HypEncoder edge_feat_norm norm min: {edge_feat_mean_norm.min().item():.4f}, After HypEncoder norm max: {edge_feat_mean_norm.max().item():.4f}, After HypEncoder  norm mean: {edge_feat_mean_norm.mean().item():.4f}")
                    to_log.update({
                        "train/after_HypEncoder_edge_feat_norm_min": edge_feat_mean_norm.min().item(),
                        "train/after_HypEncoder_edge_feat_norm_max": edge_feat_mean_norm.max().item(),
                        "train/after_HypEncoder_edge_feat_norm_mean": edge_feat_mean_norm.mean().item()
                    })
                    edge_feat_var_norm = torch.norm(edge_feat_var, p=2, dim=-1)
                    print(
                        f"After HypEncoder edge_feat_var_norm norm min: {edge_feat_var_norm.min().item():.4f}, After HypEncoder edge_feat_var_norm norm max: {edge_feat_var_norm.max().item():.4f}, After HypEncoder edge_feat_var_norm norm mean: {edge_feat_var_norm.mean().item():.4f}")
                    to_log.update({
                        "train/after_HypEncoder_edge_feat_var_norm_min": edge_feat_var_norm.min().item(),
                        "train/after_HypEncoder_edge_feat_var_norm_max": edge_feat_var_norm.max().item(),
                        "train/after_HypEncoder_edge_feat_var_norm_mean": edge_feat_var_norm.mean().item()
                    })
                else:
                    edge_feat_mean = None
                    edge_feat_var = None
                wandb.log(to_log)
                return node_feat_mean, node_feat_var, edge_feat_mean, edge_feat_var, z

    def codebook(self, feat, codebook_type="euc", node_mask=None):
        if codebook_type == "euc" and self.euc_channels > 0:
            quantize, vq_ind, vq_loss, perplexity = self.euc_codebook(
                feat, node_mask)
        elif codebook_type == "hyp" and self.hyp_channels > 0:
            quantize, vq_ind, vq_loss, perplexity = self.hyp_codebook(
                feat, node_mask)
        return quantize, vq_ind, vq_loss, perplexity

    def reparameterize(self, mean, log_std):
        try:
            # Add numerical stability checks for mean
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                print("Warning: NaN or Inf detected in mean during reparameterization")
                mean = torch.where(
                    torch.isnan(mean) | torch.isinf(mean),
                    torch.zeros_like(mean),
                    mean
                )

            # Add numerical stability checks for log_std
            if torch.isnan(log_std).any() or torch.isinf(log_std).any():
                print(
                    "Warning: NaN or Inf detected in log_std during reparameterization")
                log_std = torch.where(
                    torch.isnan(log_std) | torch.isinf(log_std),
                    torch.ones_like(log_std) * -5,  # Small but valid log std
                    log_std
                )

            # Clamp log_std to avoid extreme values
            log_std = torch.clamp(log_std, min=-20, max=2)

            # Standard normal sampling with numerical stability
            std = torch.exp(0.5 * log_std)

            # Handle potential infs or nans in std
            if torch.isnan(std).any() or torch.isinf(std).any():
                print("Warning: NaN or Inf detected in std after exp")
                std = torch.where(
                    torch.isnan(std) | torch.isinf(std),
                    torch.ones_like(std) * 0.01,  # Small but valid std
                    std
                )

            # Sample from standard normal and scale
            eps = torch.randn_like(std)
            z = mean + eps * std

            # Final safety check on output
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("Warning: NaN or Inf detected in reparameterization output")
                z = torch.where(
                    torch.isnan(z) | torch.isinf(z),
                    mean.detach(),  # Fall back to mean if sampling produces NaN/Inf
                    z
                )

            return z

        except Exception as e:
            print(f"Error in reparameterize: {e}")
            # In case of unexpected error, return mean as fallback
            return mean.detach()

    def decode(self, z, adj, mask=None):
        if self.euc_channels == 0 and self.cfg.model.transformer_decoder.Hypformer_use:
            z = self.trans_decoder(z, mask)
        if self.euc_channels > 0:
            z_euc = z[..., :self.euc_channels]
            euc2node_feat = self.euc2node_decoder(z_euc)
            euc2edge_feat = self.euc2edge_decoder(z_euc)
        else:
            z_euc = None
            euc2node_feat = None
            euc2edge_feat = None

        if self.hyp_channels > 0:
            z_hyp = z[..., self.euc_channels:]
            hyp2node_feat = self.hyp2node_decoder(z_hyp)
            hyp2edge_feat = self.hyp2edge_decoder(z_hyp)
        else:
            z_hyp = None
            hyp2node_feat = None
            hyp2edge_feat = None

        # pdb.set_trace()
        return euc2node_feat, hyp2node_feat, euc2edge_feat, hyp2edge_feat, adj
