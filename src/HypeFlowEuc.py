"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, List, Literal
import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm

from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds.stereographic import manifold
from torchmetrics import MeanMetric, MinMetric
import pytorch_lightning as pl
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint
from models.arch import tMLP, ProjectToTangent, Unbatch
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
import manifolds
from manifolds.geodesic import geodesic
from solvers import projx_integrator_return_last, projx_integrator
from src.models.hyper_vae import HypFormer as HypVAE
from src.models.hyper_vae import HypAttention
from src.models.transormer_layers import DiTAttention
from src.models.PoincareTransformer import PoincareTransformer
from src.models.TimedPoincareTransformer import TimedPoincareTransformer, FlowMapTimedPoincareTransformer
from src.distribution.wrapped_normal import WrappedNormal, WrappedNormalPoincare
import pdb
import utils
import torch.nn as nn
import math
import time
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.product import ProductManifold
import wandb
from geoopt.optim.radam import RiemannianAdam
from geoopt import ManifoldParameter

def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class ManifoldFMLitModule(pl.LightningModule):
    def __init__(self, cfg, sampling_metrics, glob_cfg):
        super().__init__()
        self.cfg = cfg
        self.glob_cfg = glob_cfg
        
        # 1. Parse the base manifold class name from config
        self.manifold_name = cfg.model.manifold
        self.manifold = eval(self.manifold_name)()

        # 2. Track original checkpoint configuration details for the graph loaders
        self.shuffle_dense_graph = glob_cfg.model.shuffle_dense_graph
        self.euc_channels = glob_cfg.model.euc_channels
        self.hyp_channels = glob_cfg.model.hyp_channels
        self.nfe_steps = glob_cfg.flow_train.integrate.num_steps
        self.reconstruct_metrics = None

        # 3. FORCE the Flow integration space to use flat Euclidean math
        # This fixes the VAE loading crash while keeping flow trajectories flat!
        self.product_manifold = Euclidean()
        print("Bypassing Stereographic Math: Euclidean Flow Manifold Assigned.")
        
        # 4. Route the Attention Network Architecture Backbone
        # We keep the TimedPoincareTransformer because your VAE checkpoint has 64 hyperbolic channels,
        # but because product_manifold is Euclidean, the vector flows will calculate straight lines.
        if self.hyp_channels > 0:
            from .models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
            if "use_flow_map" in self.cfg.integrate.method:
                self.model = FlowMapTimedPoincareTransformer(
                    cfg, PoincareBall(), glob_cfg.model.latent_channels, 
                    glob_cfg.model.transformer_encoder.trans_num_layers, glob_cfg.model.transformer_encoder.trans_num_heads, 
                    glob_cfg.model.transformer_encoder.trans_dropout, glob_cfg.model.transformer_encoder.max_seq_len, 
                    glob_cfg.model.transformer_encoder.use_hyperbolic_attention, glob_cfg.model.transformer_encoder.attention_type, 
                    glob_cfg.model.transformer_encoder.attention_activation
                )
            else:
                self.model = TimedPoincareTransformer(
                    cfg, PoincareBall(), glob_cfg.model.latent_channels, 
                    glob_cfg.model.transformer_encoder.trans_num_layers, glob_cfg.model.transformer_encoder.trans_num_heads, 
                    glob_cfg.model.transformer_encoder.trans_dropout, glob_cfg.model.transformer_encoder.max_seq_len, 
                    glob_cfg.model.transformer_encoder.use_hyperbolic_attention, glob_cfg.model.transformer_encoder.attention_type, 
                    glob_cfg.model.transformer_encoder.attention_activation
                )
        else:
            self.model = DiTAttention(cfg, self.product_manifold)

        # 5. Initialize Optimization Tracker Metrics
        self.use_riemannian_optimizer = glob_cfg.loss.use_riemannian_optimizer
        if self.use_riemannian_optimizer:
            self.automatic_optimization = False
        self.sampling_metrics = sampling_metrics
        self.reconstruct_metrics_argmax = None
        self.name = glob_cfg.general.name + "-flow"
        
        if self.cfg.scheduler == "cosine":
            self.scheduler = utils.CosineScheduler()
        else:
            self.scheduler = utils.CondOTScheduler()
            
        self.val_counter = 0
        self.train_metrics = {"loss": MeanMetric(), "log_metric": MeanMetric(), "log_metric_mean": MeanMetric()}
        self.val_metrics = {"loss": MeanMetric(), "log_metric": MeanMetric(), "log_metric_mean": MeanMetric()}
        self.test_metrics = {"loss": MeanMetric(), "log_metric": MeanMetric(), "log_metric_mean": MeanMetric()}
        self.val_metrics_best = {"loss": MeanMetric(), "log_metric": MeanMetric(), "log_metric_mean": MeanMetric()}
        
        self.all_node_feats = []  
        self.all_edge_feats = []  
        self.cnt = 0
        
        if self.cfg.source_distribution == 'set1-0.04':
            self.prior_std = 0.5 / math.sqrt(float(self.hyp_channels + self.euc_channels))
        elif self.cfg.source_distribution == 'set2-0.4':
            self.prior_std = 0.5

    def load_VAE(self, VAE):
        self.VAE = VAE
        self.dataset_infos = self.VAE.dataset_infos
        
    @torch.no_grad()
    def encode(self, batch):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        x, E = dense_data.X, dense_data.E
        if self.shuffle_dense_graph:
            x, E, node_mask = utils.shuffle_dense_graph(x, E, node_mask)
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.VAE.compute_extra_data(noisy_data)
        
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()
        
        if self.VAE.model.use_VQVAE:
            node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)
        else:
            if not self.VAE.model.use_VAE:
                node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)
            else:
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.VAE.model.encode(x, adj, E, node_mask)
                node_feat = None
                edge_feat = None
                if self.euc_channels > 0:
                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    node_feat = euc_qz_x.rsample()
                if self.hyp_channels > 0:
                    hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                    edge_feat = hyp_qz_x.rsample()
            
        return z, node_mask, node_feat, edge_feat

    @property
    def vecfield(self):
        return self.model

    @property
    def device(self):
        return self.model.parameters().__next__().device

    @torch.no_grad()
    def sample_decode(self, n_samples, x0=None, stage="valid"):
        x1, node_mask = self.sample(n_samples, x0)
        graph_list, graph_list_argmax = self.VAE.generate_sample_from_z(x1, node_mask)
                
        current_path = os.getcwd()
        result_path = os.path.join(current_path, f'graphs/{self.name}/epoch{self.current_epoch}/')
        self.VAE.visualization_tools.visualize(result_path, graph_list, n_samples)
        self.print("Visualization complete.")
        return graph_list, graph_list_argmax

    @torch.no_grad()
    def sample(self, n_samples, x0=None):
        if x0 is None:
            max_nodes = self.dataset_infos.n_nodes.shape[-1]
            num_masked_tokens = torch.multinomial(self.dataset_infos.n_nodes, num_samples=n_samples, replacement=True)
            mask = torch.zeros((n_samples, max_nodes), device=self.device, dtype=torch.int64)
            for i, n in enumerate(num_masked_tokens):
                mask[i, :n] = 1
                
            x0 = self.product_manifold.random((n_samples, max_nodes, self.hyp_channels + self.euc_channels), device=self.device, std=self.prior_std)
            
        local_coords = self.cfg.get("local_coords", False)
        eval_projx = self.cfg.get("eval_projx", False)

        if not eval_projx and not local_coords:
            x1 = odeint(
                self.vecfield, x0, t=torch.linspace(0, 1, 2).to(self.device),
                atol=self.cfg.model.atol, rtol=self.cfg.model.rtol,
                options={"min_step": 1e-5}, node_mask=mask,
            )[-1]
        else:
            x1 = projx_integrator_return_last(
                self.product_manifold, self.vecfield, x0,
                t=torch.linspace(0, 1, self.nfe_steps).to(self.device),
                method=self.cfg.integrate.method, projx=eval_projx,
                local_coords=local_coords, pbar=True, node_mask=mask,
            )

        x1_norm = torch.norm(x1, p=2, dim=-1)
        x1_x0_norm = torch.norm(x1 - x0, p=2, dim=-1)
        x_0_norm = torch.norm(x0, p=2, dim=-1)
        
        to_log = {"epoch": self.current_epoch}
        to_log.update({"val/sampled_x1_norm_mean": x1_norm.mean().item(), "val/sampled_x0_norm_mean": x_0_norm.mean().item()})
        print(f"x1_norm mean: {x1_norm.mean().item():.4f}, x_0_norm mean: {x_0_norm.mean().item():.4f}")
        wandb.log(to_log)
        
        # Plain return statement
        return x1, mask
    
    def loss_fn(self, batch: torch.Tensor):
        return self.rfm_loss_fn(batch)

    def rfm_loss_fn(self, batch: torch.Tensor):
        if self.use_riemannian_optimizer and self.trainer.training:
            opt_euc, opt_hyp = self.optimizers()
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers] if schedulers is not None else []
            if len(schedulers) > 0 and schedulers[0] is not None:
                self.log("lr_euc", schedulers[0].get_last_lr()[0], prog_bar=True)

        if isinstance(batch, dict):
            # -------------------------------------------------------------
            # Target Trajectory Paths (Handling Evaluated Primitives Explicitly)
            # -------------------------------------------------------------
            x0 = batch["x0"]
            x1 = batch["x1"]
            from .models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
            target_manifold = PoincareBall() if self.hyp_channels > 0 else Euclidean()

            N = x1.shape[0]
            MAX_NODES = x1.shape[1]
            t = torch.rand(N).reshape(-1, 1).to(x1)

            def cond_u(x0, x1, t):
                alpha_t = self.scheduler(t).alpha_t
                shooting_tangent_vec = target_manifold.logmap(x0, x1)
                x_t = target_manifold.expmap(x0, alpha_t * shooting_tangent_vec)
                u_t = target_manifold.logmap(x_t, x1)
                return x_t, u_t
                    
            x_t_u_t = [cond_u(x0_i, x1_i, t_i) for x0_i, x1_i, t_i in zip(x0, x1, t)]
            x_t = torch.stack([xt for xt, _ in x_t_u_t], dim=0).reshape(N, MAX_NODES, -1)
            u_t = torch.stack([ut for _, ut in x_t_u_t], dim=0).reshape(N, MAX_NODES, -1)

            x_t_1 = self.vecfield(t, x_t)
            v_t = self.manifold.logmap(x_t, x_t_1)

            diff = (v_t - u_t).clone().contiguous()
            dim = x_t.shape[-1]
            loss = target_manifold.inner(x_t.reshape(-1, dim), diff.reshape(-1, dim), diff.reshape(-1, dim)).mean() / dim
            
        else:
            x1 = batch
            _, node_mask, node_feat, edge_feat = self.encode(x1)
                    
            if self.hyp_channels == 0:
                x1 = node_feat
            elif self.euc_channels == 0:
                x1 = edge_feat
            else:
                x1 = torch.concat((node_feat, edge_feat), dim=-1)
            
            x0 = self.product_manifold.random(*x1.shape, device=self.device, dtype=x1.dtype, std=self.prior_std).to(x1)
            
            N = x1.shape[0]
            MAX_NODES = x1.shape[1]
            t = torch.rand(N).reshape(-1, 1).to(x1)
            s = torch.rand(N).reshape(-1, 1).to(x1)

            def cond_u(x0, x1, t):
                alpha_t = self.scheduler(t).alpha_t
                shooting_tangent_vec = self.product_manifold.logmap(x0, x1)
                x_t = self.product_manifold.expmap(x0, alpha_t * shooting_tangent_vec)
                u_t = self.product_manifold.logmap(x_t, x1)
                return x_t, u_t

            x_t_u_t = [cond_u(x0_i, x1_i, t_i) for x0_i, x1_i, t_i in zip(x0, x1, t)]
            x_t = torch.stack([xt for xt, _ in x_t_u_t], dim=0).reshape(N, MAX_NODES, -1)
            u_t = torch.stack([ut for _, ut in x_t_u_t], dim=0).reshape(N, MAX_NODES, -1)
            u_t = self.product_manifold.proju(x_t, u_t)

            # Pre-calculate 3D mask variant ahead of condition matching forks [B, N, 1]
            mask_3d = node_mask.unsqueeze(-1) if node_mask.dim() == 2 else node_mask

            if "use_flow_map" in self.cfg.integrate.method:
                # Flow distillation logic path definitions
                x_s_u_s = [cond_u(x0_i, x1_i, s_i) for x0_i, x1_i, s_i in zip(x0, x1, s)]
                x_s = torch.stack([xs for xs, _ in x_s_u_s], dim=0).reshape(N, MAX_NODES, -1)
                
                with torch.enable_grad():
                    t_req = t.clone().detach().requires_grad_(True)
                    X_st = self.vecfield(s, t_req, x_s, mask=node_mask)
                    
                    v_tt = self.vecfield(t, t, x_t, mask=node_mask)
                    loss_fm = self.product_manifold.dist(v_tt, u_t).pow(2)
                    # FIXED: Applied 3D matrix broadcasting multiplication safely
                    loss_fm = (loss_fm * mask_3d).sum() / mask_3d.sum()

                    eps = 1e-4
                    with torch.no_grad():
                        X_st_plus = self.vecfield(s, t + eps, x_s, mask=node_mask)
                        X_st_minus = self.vecfield(s, t - eps, x_s, mask=node_mask)
                    
                    dt_X = (X_st_plus - X_st_minus) / (2 * eps)
                    v_tt_at_X = self.vecfield(t_req, t_req, X_st, mask=node_mask)
                    
                    loss_lsd = self.product_manifold.dist(dt_X, v_tt_at_X).pow(2)
                    # FIXED: Applied 3D matrix broadcasting multiplication safely
                    loss_lsd = (loss_lsd * mask_3d).sum() / mask_3d.sum()
                    loss = loss_fm + loss_lsd

            elif self.cfg.integrate.method == "vt_prediction":
                x1_pred = self.vecfield(t, x_t, mask=node_mask)
                v_t = self.product_manifold.logmap(x_t, x1_pred)
                diff = (v_t - u_t).clone().contiguous()
                dim = x_t.shape[-1]
                # FIXED: Aligned tensor layouts so standard matrix products map cleanly
                loss = (self.product_manifold.inner(x_t.reshape(-1, dim), diff.reshape(-1, dim), diff.reshape(-1, dim)) * mask_3d.reshape(-1, 1)).sum() / mask_3d.sum() / dim
                
            elif self.cfg.integrate.method == "x1_prediction":
                x1_pred = self.vecfield(t, x_t, mask=node_mask)
                diff = self.product_manifold.dist(x1_pred, x1).pow(2)
                # FIXED: Aligned tensor layouts so standard matrix products map cleanly
                loss = (diff * mask_3d).sum() / mask_3d.sum()

        if self.use_riemannian_optimizer and self.trainer.training:
            opt_euc.zero_grad()
            opt_hyp.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(optimizer=opt_euc, gradient_clip_val=self.cfg.train.manual_setting.gradient_clip_val, gradient_clip_algorithm=self.cfg.train.manual_setting.gradient_clip_algorithm)
            self.clip_gradients(optimizer=opt_hyp, gradient_clip_val=self.glob_cfg.train.clip_grad, gradient_clip_algorithm="norm")
            opt_euc.step()
            opt_hyp.step()

        return {"loss": loss, "log_metric": loss.detach().item(), "log_metric_mean": 0.0}

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.loss_fn(batch)
        if torch.isfinite(loss_dict['loss']):
            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)
                
                # FIXED: Apply the identical tensor validation guardrail loop
                if isinstance(v, torch.Tensor):
                    self.train_metrics[k].update(v.detach().cpu())
                else:
                    self.train_metrics[k].update(torch.tensor(v))
                    
            return loss_dict
        return None

    def on_train_epoch_end(self):
        for m in self.train_metrics.values(): m.reset()
        if not self.use_riemannian_optimizer: return
        schedulers = self.lr_schedulers()
        for s in ([schedulers] if not isinstance(schedulers, (list, tuple)) else schedulers):
            if s is not None: s.step()

    def validation_step(self, batch: Any, batch_idx: int):
        metrics = self.val_metrics
        loss_dict = self.loss_fn(batch)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True)
            
            # FIXED: Safely verify type allocation mapping before casting to CPU memory footprints
            if isinstance(v, torch.Tensor):
                metrics[k].update(v.detach().cpu())
            else:
                # If it's already a raw Python float primitive, update the TorchMetric tracker directly
                metrics[k].update(torch.tensor(v))
                
        return loss_dict

    def on_validation_epoch_start(self):
        if self.val_counter == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            for batch in val_loader: self.VAE.test_interpolate(batch)
            
            samples_left = self.glob_cfg.general.samples_to_generate
            samples, samples_argmax = [], []
            while samples_left > 0:
                bs = self.glob_cfg.flow_train.batch_size.val
                to_gen = min(samples_left, bs)
                g, g_m = self.sample_decode(n_samples=to_gen, stage="valid")
                samples.extend(g); samples_argmax.extend(g_m)
                samples_left -= to_gen
            
            self.sampling_metrics.reset()
            self.reconstruct_metrics = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank, extra_name='softmax')
            self.sampling_metrics.reset()

    def on_validation_epoch_end(self):
        out = {}
        for k, val_metric in self.val_metrics.items():
            val_metric_value = val_metric.compute()
            self.val_metrics_best[k].update(val_metric_value)
            val_metric.reset()
            out[k] = val_metric_value
        self.val_counter += 1
        return out

    def configure_optimizers(self):
        if self.use_riemannian_optimizer:
            euc_params = [p for n, p in self.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]
            hyp_params = [p for n, p in self.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]
            optimizer_euc = hydra.utils.instantiate(self.cfg.optim.optimizer, params=euc_params, _convert_="partial")
            optimizer_hyp = hydra.utils.instantiate(self.cfg.optim.optimizer_hyp, params=hyp_params, _convert_="partial")
            return [{"optimizer": optimizer_euc}, {"optimizer": optimizer_hyp}]
        else:
            optimizer = hydra.utils.instantiate(self.cfg.optim.optimizer, params=self.parameters(), _convert_="partial")
            return {"optimizer": optimizer}