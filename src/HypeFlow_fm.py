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
# from src.ema import EMA
from models.arch import tMLP, ProjectToTangent, Unbatch

from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds.stereographic.manifold_new import PoincareBall
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
        self.manifold_name = cfg.model.manifold
        self.manifold = eval(self.manifold_name)()

        # Model of the vector field.
        # self.model = HypAttention(cfg)
        self.shuffle_dense_graph = glob_cfg.model.shuffle_dense_graph
        self.euc_channels = glob_cfg.model.euc_channels
        self.hyp_channels = glob_cfg.model.hyp_channels
        self.nfe_steps = glob_cfg.flow_train.integrate.num_steps
        self.reconstruct_metrics = None
        if self.euc_channels ==0:
            # self.product_manifold = ProductManifold((self.manifold, self.hyp_channels))
            self.product_manifold = self.manifold
        if self.hyp_channels ==0:
            self.product_manifold = ProductManifold((Euclidean(), self.euc_channels))
        if self.hyp_channels >0 and self.euc_channels >0:
            self.product_manifold = ProductManifold(
                (Euclidean(), self.euc_channels), (self.manifold, self.hyp_channels)
            )
        
        if self.glob_cfg.dataset.name in ['hyperbolic'] or self.euc_channels==0 :
            from .models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
            if "use_flow_map" in self.cfg.integrate.method:
                self.model = FlowMapTimedPoincareTransformer(cfg, PoincareBall(), glob_cfg.model.latent_channels, glob_cfg.model.transformer_encoder.trans_num_layers, glob_cfg.model.transformer_encoder.trans_num_heads, glob_cfg.model.transformer_encoder.trans_dropout, 
                                                    glob_cfg.model.transformer_encoder.max_seq_len, glob_cfg.model.transformer_encoder.use_hyperbolic_attention, glob_cfg.model.transformer_encoder.attention_type, glob_cfg.model.transformer_encoder.attention_activation)
            else:
                self.model = TimedPoincareTransformer(cfg, PoincareBall(), glob_cfg.model.latent_channels, glob_cfg.model.transformer_encoder.trans_num_layers, glob_cfg.model.transformer_encoder.trans_num_heads, glob_cfg.model.transformer_encoder.trans_dropout, 
                                                    glob_cfg.model.transformer_encoder.max_seq_len, glob_cfg.model.transformer_encoder.use_hyperbolic_attention, glob_cfg.model.transformer_encoder.attention_type, glob_cfg.model.transformer_encoder.attention_activation)
        else:
            self.model = DiTAttention(cfg, self.product_manifold)

        
        # getattr(models, self.manifold_name)()
         
        # how to use sampling_metrics?
        self.use_riemannian_optimizer = glob_cfg.loss.use_riemannian_optimizer
        if self.use_riemannian_optimizer:
            self.automatic_optimization = False
        self.sampling_metrics = sampling_metrics
        self.reconstruct_metrics_argmax = None
        self.name = glob_cfg.general.name + "-flow"
        if self.cfg.scheduler=="cosine":
            self.scheduler = utils.CosineScheduler()
        else:
            self.scheduler = utils.CondOTScheduler()
        self.val_counter = 0
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        self.val_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        self.test_metrics = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }

        # for logging best so far validation accuracy
        self.val_metrics_best = {
            "loss": MeanMetric(),
            "log_metric": MeanMetric(),
            "log_metric_mean": MeanMetric()
        }
        
        self.all_node_feats = []  # 收集所有有效的 node_feat
        self.all_edge_feats = []  # 收集所有有效的 edge_feat
        self.cnt = 0
        if self.cfg.source_distribution == 'set1-0.04':
            self.prior_std = 0.5 / math.sqrt(float(self.hyp_channels+self.euc_channels))
        elif self.cfg.source_distribution == 'set2-0.4':
            self.prior_std = 0.5

    # add a function to load encoder+
    def load_VAE(self, VAE):
        self.VAE = VAE
        self.dataset_infos =  self.VAE.dataset_infos
        
    @torch.no_grad()
    def encode(self, batch):
        dense_data, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # dense_data = dense_data.mask(node_mask)
        x, E = dense_data.X, dense_data.E
        if self.shuffle_dense_graph:
            x, E, node_mask = utils.shuffle_dense_graph(x, E, node_mask)
        adj, edge_labels = utils.process_edge_attr(E, node_mask)
        noisy_data = {'X': x, 'E': edge_labels, 'y': None, 'node_mask': node_mask}
        extra_data = self.VAE.compute_extra_data(noisy_data)
        # x = torch.cat((x, extra_data.X), dim=2).float()
        y_repeated = extra_data.y.unsqueeze(1).repeat(1, extra_data.X.size(1), 1)
        x = torch.cat((x, extra_data.X, y_repeated), dim=2).float()
        E = torch.cat((edge_labels, extra_data.E), dim=3).float()
        # VQ VAE case
        if self.VAE.model.use_VQVAE:
            node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)

        else:
            # AE case
            if not self.VAE.model.use_VAE:
                node_feat, edge_feat, z = self.VAE.model.encode(x, adj, E, node_mask)
                # pdb.set_trace()
            # VAE case  
            else:
                euc_feat_mean, euc_feat_logvar, hyp_feat_mean, hyp_feat_logvar, z = self.VAE.model.encode(x, adj, E, node_mask)
                # pdb.set_trace()
                node_feat=None
                edge_feat=None
                if self.euc_channels>0:
                    euc_qz_x = torch.distributions.Normal(euc_feat_mean, torch.exp(0.5 * euc_feat_logvar))
                    node_feat = euc_qz_x.rsample()
                if self.hyp_channels>0:
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
    def visualize(self, batch, force=False):
        if not force and not self.cfg.get("visualize", False):
            return

        if isinstance(self.manifold, Sphere) and self.euc_channels == 3:
            self.plot_earth2d(batch)

        if isinstance(self.manifold, FlatTorus) and self.euc_channels == 2:
            self.plot_torus2d(batch)

        if isinstance(self.manifold, Mesh) and self.euc_channels == 3:
            self.plot_mesh(batch)

        if isinstance(self.manifold, SPD) and self.euc_channels >= 3:
            self.plot_spd(batch)

        if isinstance(self.manifold, PoincareBall) and self.euc_channels == 2:
            self.plot_poincare(batch)

    @torch.no_grad()
    def plot_poincare(self, batch):
        os.makedirs("figs", exist_ok=True)

        x0 = batch["x0"]
        x1 = batch["x1"]

        trajs = self.sample_all(x1.shape[0], device=x1.device, x0=x0)
        samples = trajs[-1]

        # Plot model samples
        x0 = x0.detach().cpu().numpy()
        x1 = x1.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        trajs = trajs.detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=2, color="C3")
        plt.gca().add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"figs/samples-{self.global_step:06d}.png")
        plt.savefig(f"figs/samples-{self.global_step:06d}.pdf")
        plt.close()

        # Plot trajectories
        plt.figure(figsize=(6, 6))
        plt.scatter(x0[:, 0], x0[:, 1], s=2, color="C0")
        plt.scatter(x1[:, 0], x1[:, 1], s=2, color="C1")
        for i in range(100):
            plt.plot(trajs[:, i, 0], trajs[:, i, 1], color="grey", linewidth=0.5)
        plt.gca().add_patch(plt.Circle((0, 0), 1.0, color="k", fill=False))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"figs/trajs-{self.global_step:06d}.png")
        plt.savefig(f"figs/trajs-{self.global_step:06d}.pdf")
        plt.close()


    @torch.no_grad()
    def compute_cost(self, batch):
        if isinstance(batch, dict):
            x0 = batch["x0"]
        else:
            x0 = (
                self.manifold.random_base(batch.shape[0], self.euc_channels)
                .reshape(batch.shape[0], self.euc_channels)
                .to(batch.device)
            )

        # Solve ODE.
        x1 = odeint(
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 2).to(x0.device),
            atol=self.cfg.model.atol,
            rtol=self.cfg.model.rtol,
        )[-1]

        x1 = self.manifold.projx(x1)

        return self.manifold.dist(x0, x1)

    @torch.no_grad()
    def sample_decode(self, n_samples, x0=None, stage="valid"):
        x1, node_mask = self.sample(n_samples, x0)
        graph_list, graph_list_argmax = self.VAE.generate_sample_from_z(x1, node_mask)
                
        # Only once after full pass
        current_path = os.getcwd()
        result_path = os.path.join(
            current_path,
            f'graphs/{self.name}/epoch{self.current_epoch}/'
        )
        self.VAE.visualization_tools.visualize(result_path, graph_list, n_samples)
        self.print("Visualization complete.")
        
        return graph_list, graph_list_argmax

    @torch.no_grad()
    def sample(self, n_samples, x0=None):
        if x0 is None:
            # Sample from base distribution.
            max_nodes = self.dataset_infos.n_nodes.shape[-1] # may be we need max_node = max_nodes - 1
            num_masked_tokens = torch.multinomial(self.dataset_infos.n_nodes, num_samples=n_samples, replacement=True)
            mask = torch.zeros((n_samples, max_nodes), device=self.device, dtype=torch.int64)
            for i, n in enumerate(num_masked_tokens):
                mask[i, :n] = 1
            # mask = mask.float()
            # x0 = self.product_manifold.random((n_samples, max_nodes, self.hyp_channels+self.euc_channels), device=self.device, std=self.prior_std)
            # pdb.set_trace()
            # if self.VAE.model.hyp_codebook is not None:
            #     hyp_indices = torch.randint(0, self.VAE.model.hyp_codebook.codebook.shape[0], (n_samples, max_nodes), device=self.device)
            #     x0 = F.embedding(hyp_indices, self.VAE.model.hyp_codebook.codebook)
            # else:
            #     x0 = self.product_manifold.random((n_samples, max_nodes, self.hyp_channels+self.euc_channels), device=self.device, std=self.prior_std)
            x0 = self.product_manifold.random((n_samples, max_nodes, self.hyp_channels+self.euc_channels), device=self.device, std=self.prior_std)
            
        local_coords = self.cfg.get("local_coords", False)
        eval_projx = self.cfg.get("eval_projx", False)

        # Solve ODE.
        if not eval_projx and not local_coords:
            # If no projection, use adaptive step solver.
            x1 = odeint(
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, 2).to(self.device),
                atol=self.cfg.model.atol,
                rtol=self.cfg.model.rtol,
                options={"min_step": 1e-5},
                node_mask = mask,
            )[-1]
        else:
            # If projection, use 1000 steps.
            x1 = projx_integrator_return_last(
                self.product_manifold,
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, self.nfe_steps).to(self.device),
                method=self.cfg.integrate.method,
                projx=eval_projx,
                local_coords=local_coords,
                pbar=True,
                node_mask = mask,
            )
            # x1 = projx_integrator_return_last(
            #     self.product_manifold,
            #     self.vecfield,
            #     x0,
            #     t=torch.linspace(0, 1, self.nfe_steps).to(self.device),
            #     # method="euler",
            #     method="x0_pred",
            #     projx=eval_projx,
            #     local_coords=local_coords,
            #     pbar=True,
            #     node_mask = mask,
            # )
        # x1 = self.manifold.projx(x1)
        x1_norm = torch.norm(x1, p=2, dim=-1)
        x1_x0_norm = torch.norm(x1 - x0, p=2, dim=-1)
        x_0_norm = torch.norm(x0, p=2, dim=-1)
        to_log = {"epoch": self.current_epoch}
        to_log.update({"val/sampled_x1_norm_min": x1_norm.min().item()})
        to_log.update({"val/sampled_x1_norm_max": x1_norm.max().item()})
        to_log.update({"val/sampled_x1_norm_mean": x1_norm.mean().item()})
        to_log.update({"val/sampled_x1_x0_norm_min": x1_x0_norm.min().item()})
        to_log.update({"val/sampled_x1_x0_norm_max": x1_x0_norm.max().item()})
        to_log.update({"val/sampled_x1_x0_norm_mean": x1_x0_norm.mean().item()})
        to_log.update({"val/sampled_x0_norm_min": x_0_norm.min().item()})
        to_log.update({"val/sampled_x0_norm_max": x_0_norm.max().item()})
        to_log.update({"val/sampled_x0_norm_mean": x_0_norm.mean().item()})
        print(f"x1_norm min: {x1_norm.min().item():.4f}, x1_norm max: {x1_norm.max().item():.4f}, x1_norm mean: {x1_norm.mean().item():.4f}")
        print(f"x1_x0_norm min: {x1_x0_norm.min().item():.4f}, x1_x0_norm max: {x1_x0_norm.max().item():.4f}, x1_x0_norm mean: {x1_x0_norm.mean().item():.4f}")
        print(f"x_0_norm min: {x_0_norm.min().item():.4f}, x_0_norm max: {x_0_norm.max().item():.4f}, x_0_norm mean: {x_0_norm.mean().item():.4f}")
        wandb.log(to_log)
        
        return x1, mask

    @torch.no_grad()
    def sample_all(self, n_samples, device, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.euc_channels)
                .reshape(n_samples, self.euc_channels)
                .to(device)
            )

        # Solve ODE.
        xs, _ = projx_integrator(
            self.product_manifold,
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, self.nfe_steps).to(device),
            method="euler",
            projx=True,
            pbar=True,
        )
        return xs

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: torch.Tensor,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps=1000,
    ):
        """Computes the negative log-likelihood of a batch of data."""

        try:
            nfe = [0]

            div_mode = self.cfg.get("div_mode", "exact")

            with torch.inference_mode(mode=False):
                v = None
                if div_mode == "rademacher":
                    v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1

                def odefunc(t, tensor):
                    nfe[0] += 1
                    t = t.to(tensor)
                    x = tensor[..., : self.euc_channels]
                    vecfield = lambda x: self.vecfield(t, x)
                    dx, div = output_and_div(vecfield, x, v=v, div_mode=div_mode)

                    if hasattr(self.manifold, "logdetG"):

                        def _jvp(x, v):
                            return jvp(self.manifold.logdetG, (x,), (v,))[1]

                        corr = vmap(_jvp)(x, dx)
                        div = div + 0.5 * corr.to(div)

                    div = div.reshape(-1, 1)
                    del t, x
                    return torch.cat([dx, div], dim=-1)

                # Solve ODE on the product manifold of data manifold x euclidean.
                product_man = ProductManifold(
                    (self.manifold, self.euc_channels), (Euclidean(), 1)
                )
                state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)

                local_coords = self.cfg.get("local_coords", False)
                eval_projx = self.cfg.get("eval_projx", False)

                with torch.no_grad():
                    if not eval_projx and not local_coords:
                        # If no projection, use adaptive step solver.
                        state0 = odeint(
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, 2).to(batch),
                            atol=self.cfg.model.atol,
                            rtol=self.cfg.model.rtol,
                            method="dopri5",
                            options={"min_step": 1e-5},
                        )[-1]
                    else:
                        # If projection, use 1000 steps.
                        state0 = projx_integrator_return_last(
                            product_man,
                            odefunc,
                            state1,
                            t=torch.linspace(t1, 0, num_steps + 1).to(batch),
                            method="euler",
                            projx=eval_projx,
                            local_coords=local_coords,
                            pbar=True,
                        )

                # log number of function evaluations
                self.log("nfe", nfe[0], prog_bar=True, logger=True)

                x0, logdetjac = state0[..., : self.euc_channels], state0[..., -1]
                x0_ = x0
                x0 = self.manifold.projx(x0)

                # log how close the final solution is to the manifold.
                integ_error = (x0[..., : self.euc_channels] - x0_[..., : self.euc_channels]).abs().max()
                self.log("integ_error", integ_error)

                logp0 = self.manifold.base_logprob(x0)
                logp1 = logp0 + logdetjac

                if self.cfg.get("normalize_loglik", False):
                    logp1 = logp1 / self.euc_channels

                # Mask out those that left the manifold
                masked_logp1 = logp1
                if isinstance(self.manifold, SPD):
                    mask = integ_error < 1e-5
                    self.log("frac_within_manifold", mask.sum() / mask.nelement())
                    masked_logp1 = logp1[mask]

                if return_projx_error:
                    return logp1, integ_error
                else:
                    return masked_logp1
        except:
            traceback.print_exc()
            return torch.zeros(batch.shape[0]).to(batch)

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
            if len(schedulers) > 1 and schedulers[1] is not None:
                self.log("lr_hyp", schedulers[1].get_last_lr()[0], prog_bar=False)
        # import pdb; pdb.set_trace()
        if isinstance(batch, dict):
            x0 = batch["x0"]
            x1 = batch["x1"]
            from .models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
            self.hyp_manifold = PoincareBall()

            N = x1.shape[0]
            MAX_NODES = x1.shape[1]
            t = torch.rand(N).reshape(-1, 1).to(x1)
            # unsqueeze(2).repeat(1, MAX_NODES, 1).
            def cond_u(x0, x1, t):
                alpha_t = self.scheduler(t).alpha_t
                shooting_tangent_vec = self.hyp_manifold.logmap(x0, x1)
                x_t = self.hyp_manifold.expmap(x0, alpha_t * shooting_tangent_vec)
                u_t = self.hyp_manifold.logmap(x_t, x1)
                return x_t, u_t
                    

            # Avoid functorch vmap here: geoopt logmap/expmap can fail on BatchedTensor.
            x_t_u_t = [cond_u(x0_i, x1_i, t_i) for x0_i, x1_i, t_i in zip(x0, x1, t)]
            x_t = torch.stack([xt for xt, _ in x_t_u_t], dim=0)
            u_t = torch.stack([ut for _, ut in x_t_u_t], dim=0)
            x_t = x_t.reshape(N, MAX_NODES, -1)
            u_t = u_t.reshape(N, MAX_NODES, -1)

            # x_t = self.vecfield(t, x_t).squeeze(0)
            # v_t = self.manifold.logmap(x0, x_t)
            x_t_1 = self.vecfield(t, x_t)
            v_t = self.manifold.logmap(x_t, x_t_1)

            diff = (v_t - u_t).clone().contiguous()
            dim = x_t.shape[-1]
            # loss = (torch.norm(diff, p=2, dim=-1) ** 2).mean() / dim
            loss = self.hyp_manifold.inner(x_t.reshape(-1, dim), 
                                           diff.reshape(-1, dim), 
                                           diff.reshape(-1, dim)).mean() / dim

            
        else:
            x1 = batch
            _, node_mask, node_feat, edge_feat = self.encode(x1)
                    
            if self.hyp_channels==0:
                x1 = node_feat
            if self.euc_channels==0:
                x1 = edge_feat
            if self.hyp_channels>0 and self.euc_channels>0:
                x1 = torch.concat((node_feat, edge_feat), dim=-1)
            
            
            # x0 = self.product_manifold.random(*x1.shape, device=self.device, dtype=x1.dtype, std=self.prior_std).to(x1)

            # 从码本的大小中采一个均匀分布的index, 然后根据这个index从码本中采样一个向量作为x0
            # pdb.set_trace()
            # if self.VAE.model.hyp_codebook is not None:
            #     hyp_indices = torch.randint(0, self.VAE.model.hyp_codebook.codebook.shape[0], (x1.shape[0],x1.shape[1], ), device=self.device)
            #     x0 = F.embedding(hyp_indices, self.VAE.model.hyp_codebook.codebook)
            # else:
            #     x0 = self.product_manifold.random(*x1.shape, device=self.device, dtype=x1.dtype, std=self.prior_std).to(x1)
            x0 = self.product_manifold.random(*x1.shape, device=self.device, dtype=x1.dtype, std=self.prior_std).to(x1)
            
            
            # log distribution of x0
            x0_norm = torch.norm(x0, p=2, dim=-1)
            x1_norm = torch.norm(x1, p=2, dim=-1)
            print(f"x1_norm min: {x1_norm.min().item():.4f}, x1_norm max: {x1_norm.max().item():.4f}, x1_norm mean: {x1_norm.mean().item():.4f}")
            print(f"x0_norm min: {x0_norm.min().item():.4f}, x0_norm max: {x0_norm.max().item():.4f}, x0_norm mean: {x0_norm.mean().item():.4f}")
            to_log = {"epoch": self.current_epoch}
            to_log.update({"train/x0_norm_min": x0_norm.min().item()})
            to_log.update({"train/x0_norm_max": x0_norm.max().item()})
            to_log.update({"train/x0_norm_mean": x0_norm.mean().item()})
            to_log.update({"train/x1_norm_min": x1_norm.min().item()})
            to_log.update({"train/x1_norm_max": x1_norm.max().item()})
            to_log.update({"train/x1_norm_mean": x1_norm.mean().item()})
            wandb.log(to_log)

            N = x1.shape[0]
            MAX_NODES = x1.shape[1]
            t = torch.rand(N).reshape(-1, 1).to(x1)
            s = torch.rand(N).reshape(-1, 1).to(x1) * t
            s.clamp_(min=torch.tensor(0.0, device=self.device), max=t - 1e-7) ##TODO: replace this by taking in as config

            # unsqueeze(2).repeat(1, MAX_NODES, 1).
            def cond_u(x0, x1, t):
                alpha_t = self.scheduler(t).alpha_t
                shooting_tangent_vec = self.product_manifold.logmap(x0, x1)
                
                x_t = self.product_manifold.expmap(x0, alpha_t * shooting_tangent_vec)
                u_t = self.product_manifold.logmap(x_t, x1)
                return x_t, u_t

            # Avoid functorch vmap here: geoopt logmap/expmap can fail on BatchedTensor.
            x_t_u_t = [cond_u(x0_i, x1_i, t_i) for x0_i, x1_i, t_i in zip(x0, x1, t)]
            x_t = torch.stack([xt for xt, _ in x_t_u_t], dim=0)
            u_t = torch.stack([ut for _, ut in x_t_u_t], dim=0)
            x_t = x_t.reshape(N, MAX_NODES, -1)
            u_t = u_t.reshape(N, MAX_NODES, -1)
            u_t = self.product_manifold.proju(x_t, u_t)

            if "use_flow_map" in self.cfg.integrate.method:
                with torch.no_grad():
                    sd_weight = (t - s)
                    # Shape: (N, 1, 1) to broadcast over (N, MAX_NODES, D)
                    sd_weight = sd_weight.view(-1, *[1] * (len(x1.shape) - 1))

                # LSD loss — already masked inside _lsd_loss
                lsd = self._lsd_loss(x0, x1, s, t, node_mask=node_mask)
                loss = (sd_weight.squeeze() * lsd).sum() / node_mask.sum() \
                    if node_mask is not None else (sd_weight * lsd).mean()

                # Flow matching distillation target
                with torch.no_grad():
                    xt, vt = self.product_manifold.geodesic_with_tangent(x0, x1, t)

                out = self.vecfield(t, t, xt, mask=node_mask)
                diff = out - vt
                ## swap the inner for manual calculation
                fm_loss_per_node = self.product_manifold.inner(
                    xt.reshape(-1, xt.shape[-1]),
                    diff.reshape(-1, diff.shape[-1]),
                    diff.reshape(-1, diff.shape[-1])
                ).reshape(x1.shape[0], x1.shape[1])  # (N, MAX_NODES)

                if node_mask is not None:
                    fm_loss = (fm_loss_per_node * node_mask).sum() / node_mask.sum()
                else:
                    fm_loss = fm_loss_per_node.mean()

                loss = loss + fm_loss

            elif self.cfg.integrate.method == "vt_prediction":
                x1_pred = self.vecfield(t, x_t, mask=node_mask)
                v_t = self.product_manifold.logmap(x_t, x1_pred)
                diff = (v_t - u_t).clone().contiguous()
                dim = x_t.shape[-1]

                loss = (self.product_manifold.inner(x_t.reshape(-1, dim), diff.reshape(-1, dim), 
                       diff.reshape(-1, dim)) * node_mask.reshape(-1, 1)).sum() / node_mask.sum() / dim
            elif self.cfg.integrate.method == "x1_prediction":
                x1_pred = self.vecfield(t, x_t, mask=node_mask)
                diff = self.product_manifold.dist(x1_pred, x1)
                diff = diff ** 2
                loss = (diff * node_mask).sum() / node_mask.sum()
            elif self.cfg.integrate.method == "rg_vfm":
                x1_pred = self.vecfield(t, x_t, mask=node_mask)
                d2 = self.product_manifold.dist(x1_pred, x1) ** 2
                loss = (d2 * node_mask).sum() / node_mask.sum()

        if self.use_riemannian_optimizer and self.trainer.training:
            opt_euc.zero_grad()
            opt_hyp.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer=opt_euc,
                gradient_clip_val=self.cfg.train.manual_setting.gradient_clip_val,
                gradient_clip_algorithm=self.cfg.train.manual_setting.gradient_clip_algorithm

            )

            # utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)
            self.clip_gradients(
                optimizer=opt_hyp,
                gradient_clip_val=self.glob_cfg.train.clip_grad,
                gradient_clip_algorithm="norm"
            )
            opt_euc.step()
            opt_hyp.step()

        log_metric = loss.detach().item()
        log_metric_mean = 0
        if self.reconstruct_metrics is not None:
            for key, value in self.reconstruct_metrics.items():
                if self.glob_cfg.dataset.name == "qm9":
                    log_metric = log_metric * (2-torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device)))
                    log_metric_mean = log_metric_mean - torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))
                else:
                    log_metric = log_metric * torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))
                    log_metric_mean = log_metric_mean + torch.abs(torch.tensor(value, dtype=torch.float32,
                                                 device=self.device))

        return {
            "loss": loss,
            "log_metric": log_metric,
            "log_metric_mean": log_metric_mean
        }
        
    def _xst(self, i_s, s, t, node_mask=None):
        """
        Computes xst through reparameterisation.
        """
        vecfield_out = self.vecfield(s, t, i_s, mask=node_mask)
        
        # t and s are (N, 1); vecfield_out is (N, MAX_NODES, D)
        # We need (t - s) to broadcast as (N, 1, 1)
        dt = (t - s).view(t.shape[0], *([1] * (vecfield_out.dim() - 1)))
        
        return self.product_manifold.exp(
            i_s,
            dt * vecfield_out,
        )
        
    def _interpolant(self, x0, x1, t):
        return self.manifold.geodesic_interpolant(x0, x1, t)
    
    def _lsd_loss(self, x0, x1, s, t, node_mask=None):
        eps = 1e-3

        with torch.no_grad():
            i_s = self._interpolant(x0, x1, s)

        t_plus  = (t + eps).clamp(max=1.0)
        t_minus = (t - eps).clamp(min=0.0)

        with torch.no_grad():
            xst       = self._xst(i_s, s, t,       node_mask=node_mask)
            xst_plus  = self._xst(i_s, s, t_plus,  node_mask=node_mask)
            xst_minus = self._xst(i_s, s, t_minus, node_mask=node_mask)

        actual_step = (t_plus - t_minus).view(-1, *[1] * (len(xst.shape) - 1))
        dvdt = (xst_plus - xst_minus) / actual_step.clamp_min(1e-8)

        xst = xst.detach()

        with torch.no_grad():
            ref = self.vecfield(t, t, xst, mask=node_mask)

        diff = ref - dvdt  # (N, MAX_NODES, D)

        N, MAX_NODES, D = xst.shape  # 32, 17, 64


        x_sq_norm = xst.pow(2).sum(dim=-1, keepdim=True).clamp(max=1 - 1e-4)  # (N, MAX_NODES, 1)
        lambda_x = 2.0 / (1.0 - x_sq_norm)                                    # (N, MAX_NODES, 1)
        
        # Squared Riemannian norm per node: (N, MAX_NODES)
        loss_per_node = (lambda_x ** 2) * (diff ** 2).sum(dim=-1, keepdim=True)
        loss_per_node = loss_per_node.squeeze(-1)                              # (N, MAX_NODES)

        if node_mask is not None:
            loss = (loss_per_node * node_mask).sum() / node_mask.sum()
        else:
            loss = loss_per_node.mean()

        return loss
    
    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.loss_fn(batch)

        if torch.isfinite(loss_dict['loss']):
            # log train metrics
            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)
                self.train_metrics[k].update(v.cpu())
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss_dict['loss'].item()}.")
            return None
        return loss_dict

    def on_train_epoch_end(self):
        for train_metric in self.train_metrics.values():
            train_metric.reset()

        if not self.use_riemannian_optimizer:
            return

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers] if schedulers is not None else []

        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()

    def shared_eval_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
        compute_loss: bool,
        compute_nll: bool,
    ) -> dict[str, torch.Tensor]:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        out = {}
        if compute_loss:
            loss_dict = self.loss_fn(batch)
            for k, v in loss_dict.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_epoch=True,
                    prog_bar=True,
                    # batch_size=batch.batch.max().item()+1,
                )
                metrics[k].update(v.cpu())
            out.update(loss_dict)

        if compute_nll:
            if isinstance(batch, dict):
                batch = batch["x1"]
            nll_dict = {}
            logprob = self.compute_exact_loglikelihood(
                batch,
                stage,
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )
            nll = -logprob.mean()
            nll_dict[f"{stage}/nll"] = nll
            nll_dict[f"{stage}/nll_num_steps"] = self.cfg.integrate.num_steps
            self.logger.log_metrics(nll_dict, step=self.global_step)
            metrics["nll"].update(nll.cpu())
            out.update(nll_dict)

        return out
    
    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="val",
            compute_loss=True,
            compute_nll=self.cfg.val.compute_nll,
        )

    def compute_and_log_sampling_metrics(self, valid_result):
        dataset_name = self.glob_cfg.dataset.name
        base_statistics = {'sbm': [0.0008, 0.0332, 0.0255], 'planar': [0.0002, 0.0310, 0.0005], 'community': [0.02, 0.07, 0.01]}
        
        # Check if we have base statistics for this dataset
        if dataset_name in base_statistics:
            base_statistics = base_statistics[dataset_name]
            # Process datasets with base statistics for normalization
            degree_dist, clustering_dist, orbit_dist = valid_result['degree'], valid_result['clustering'], valid_result['orbit']
            degree_dist = degree_dist / base_statistics[0]
            clustering_dist = clustering_dist / base_statistics[1]
            orbit_dist = orbit_dist / base_statistics[2]
            print('degree dist: {:.3f}'.format(degree_dist))
            print('clustering dist: {:.3f}'.format(clustering_dist))
            print('orbit dist: {:.3f}'.format(orbit_dist))

            print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()
                                       
            valid_result["ratio_degree"] = degree_dist
            valid_result["ratio_clustering"] = clustering_dist
            valid_result["ratio_orbit_dist"] = orbit_dist
        else:
            # For datasets without base statistics, log the raw metrics
            print(f"No base statistics for {dataset_name}, using raw metrics")
            
            if 'degree' in valid_result:
                print('Raw degree dist: {:.3f}'.format(valid_result['degree']))
            if 'clustering' in valid_result:
                print('Raw clustering dist: {:.3f}'.format(valid_result['clustering']))
            if 'orbit' in valid_result:
                print('Raw orbit dist: {:.3f}'.format(valid_result['orbit']))
            
            if 'sampling/frac_unique' in valid_result:
                print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            if 'sampling/frac_unique_non_iso' in valid_result:
                print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            if 'sampling/frac_unic_non_iso_valid' in valid_result:
                print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()


        abstract_dataset = ['sbm', 'planar', 'community']
        
        if dataset_name in abstract_dataset:
            degree_dist, clustering_dist, orbit_dist = valid_result['degree'], valid_result['clustering'], valid_result['orbit']
            degree_dist = degree_dist / base_statistics[0]
            clustering_dist = clustering_dist / base_statistics[1]
            orbit_dist = orbit_dist / base_statistics[2]
            print('degree dist: {:.3f}'.format(degree_dist))
            print('clustering dist: {:.3f}'.format(clustering_dist))
            print('orbit dist: {:.3f}'.format(orbit_dist))

            print('Unique: {:.3f}'.format(valid_result['sampling/frac_unique']))
            print('Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unique_non_iso']))
            print('Valid&Unique&Novel: {:.3f}'.format(valid_result['sampling/frac_unic_non_iso_valid']))
            print()
                                       
            valid_result["ratio_degree"] = degree_dist
            valid_result["ratio_clustering"] = clustering_dist
            valid_result["ratio_orbit_dist"] = orbit_dist
        
        for metric_name, value in valid_result.items():
            print(f'{metric_name}: {value:.3f}')
            self.log(f'sampling_metrics/{metric_name}', value, on_epoch=True, prog_bar=True)


    
    def on_validation_epoch_start(self):
        if self.val_counter == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            # recon_samples = []
            # get first batch
            for batch in val_loader:
                self.VAE.test_interpolate(batch)
            # pdb.set_trace()
            start = time.time()
            samples_left_to_generate = self.glob_cfg.general.samples_to_generate
            samples = []
            samples_argmax = []
            while samples_left_to_generate > 0:
                bs = self.glob_cfg.flow_train.batch_size.val
                to_generate = min(samples_left_to_generate, bs)
                graph_list, graph_list_argmax = self.sample_decode(n_samples=to_generate, stage="valid")
                samples.extend(graph_list)
                samples_argmax.extend(graph_list_argmax)
                samples_left_to_generate -= to_generate
            
            self.sampling_metrics.reset()
            self.reconstruct_metrics = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank, extra_name='softmax')
            self.sampling_metrics.reset()
            self.reconstruct_metrics_argmax = self.sampling_metrics(samples_argmax, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank, extra_name='argmax')
            self.sampling_metrics.reset()

    def on_validation_epoch_end(self):
        out = {}
        for key, val_metric in self.val_metrics.items():
            val_metric_value = (
                val_metric.compute()
            )  # get val accuracy from current epoch
            val_metric_best = self.val_metrics_best[key]
            val_metric_best.update(val_metric_value)
            self.log(
                f"val/best/{key}",
                val_metric_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )
            val_metric.reset()
            out[key] = val_metric_value
            
            
        self.val_counter += 1
        if self.val_counter % self.glob_cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.glob_cfg.general.samples_to_generate
            samples = []
            samples_argmax = []
            while samples_left_to_generate > 0:
                bs = self.glob_cfg.flow_train.batch_size.val
                to_generate = min(samples_left_to_generate, bs)
                graph_list, graph_list_argmax = self.sample_decode(n_samples=to_generate, stage="test")
                samples.extend(graph_list)
                samples_argmax.extend(graph_list_argmax)
                samples_left_to_generate -= to_generate
            
            self.sampling_metrics.reset()
            self.reconstruct_metrics = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank, extra_name='softmax')
            self.sampling_metrics.reset()
            self.reconstruct_metrics_argmax = self.sampling_metrics(samples_argmax, self.name, self.current_epoch, val_counter=-1, test=False, local_rank=self.local_rank, extra_name='argmax')
            self.sampling_metrics.reset()
            
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank, extra_name='softmax')
            self.sampling_metrics.reset()
            self.sampling_metrics(samples_argmax, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank, extra_name='argmax')
            self.sampling_metrics.reset()
        return out
        
    def test_step(self, batch: Any, batch_idx: int):
        return None

    def on_test_epoch_end(self):
        for test_metric in self.test_metrics.values():
            test_metric.reset()
            
        samples_left_to_generate = self.glob_cfg.general.final_model_samples_to_generate
        samples = []
        samples_argmax = []
        while samples_left_to_generate > 0:
            bs = 512
            to_generate = min(samples_left_to_generate, bs)
            graph_list, graph_list_argmax = self.sample_decode(n_samples=to_generate, stage="test")
            samples.extend(graph_list)
            samples_argmax.extend(graph_list_argmax)
            samples_left_to_generate -= to_generate

        self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank, extra_name='softmax')
        self.sampling_metrics.reset()
        self.sampling_metrics(samples_argmax, self.name, self.current_epoch, val_counter=-1, test=True, local_rank=self.local_rank, extra_name='argmax')
        self.sampling_metrics.reset()

    def configure_optimizers(self):
        # pdb.set_trace()
        if self.use_riemannian_optimizer:
            euc_params = [p for n, p in self.named_parameters() if p.requires_grad and not isinstance(p, ManifoldParameter)]
            hyp_params = [p for n, p in self.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]

            # print hyp params name
            hyp_name = [n for n, p in self.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]
            print(hyp_name)
            optimizer_euc = hydra.utils.instantiate(
                self.cfg.optim.optimizer,
                params=euc_params,
                _convert_="partial",
            )
            optimizer_hyp = hydra.utils.instantiate(
                self.cfg.optim.optimizer_hyp,
                params=hyp_params,
                _convert_="partial",
            )

            schedulers = []
            optimizers = [{"optimizer": optimizer_euc}, {"optimizer": optimizer_hyp}]

            if self.cfg.optim.get("lr_scheduler", None) is not None:
                sched_euc = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer_euc,
                )
                sched_hyp = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer_hyp,
                )

                schedulers = [
                    {
                        "optimizer": optimizer_euc,
                        "lr_scheduler": {
                            "scheduler": sched_euc,
                            "interval": self.cfg.optim.interval,
                        }
                    },
                    {
                        "optimizer": optimizer_hyp,
                        "lr_scheduler": {
                            "scheduler": sched_hyp,
                            "interval": self.cfg.optim.interval,
                        }
                    }
                ]
                return schedulers
            return optimizers
        else:
        
            optimizer = hydra.utils.instantiate(
                self.cfg.optim.optimizer,
                params=self.parameters(),
                _convert_="partial",
            )
            if self.cfg.optim.get("lr_scheduler", None) is not None:
                lr_scheduler = hydra.utils.instantiate(
                    self.cfg.optim.lr_scheduler,
                    optimizer,
                )
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            "interval": "epoch",
                            "monitor": self.cfg.optim.monitor,
                            "frequency": self.cfg.optim.frequency,
                        },
                    }
                elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            "interval": self.cfg.optim.interval,
                        },
                    }
                else:
                    raise NotImplementedError("unsuported lr_scheduler")
            else:
                return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

def check_latent_feature():
    return NotImplementedError


def debug_u_and_x():
    return NotImplementedError