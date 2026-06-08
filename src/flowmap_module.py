"""
Generalised Flow Map module, supporting all three losses.
"""

from torch import Tensor
import torch
import utils
import os
from torch.nn import functional as F
from flow.gfm.gfm.models import RiemannianGenerativeModule, match_dims
from distribution.wrapped_normal import WrappedNormal, WrappedNormalPoincare
        
class FlowMapModule(RiemannianGenerativeModule):
    """
    Generalised Flow Map module, supporting Lagrangian, Progressive and
    Eulerian self-distillation losses.

    :param fm_loss_weight: Weighting for the flow map loss.
    :param underlying_loss: Underlying self-distillation loss to use. One of
        "esd", "lsd" or "psd".
    :param eval_nll_exact: Whether to evaluate the exact negative log-likelihood
        during validation.
    :param min_t: Minimum time for sampling t.
    :param min_gap: Minimum gap between s and t.
    :param max_t: Maximum time for sampling t.
    :param prior_args: Arguments for the prior distribution.
    :param sd_weight: Time-dependent weighting for self-distillation loss. One of
        None or "lin".
    """

    def __init__(
        self,
        fm_loss_weight: float = 1.0,
        underlying_loss: str = "esd",
        eval_nll_exact: bool = True,
        min_t: float = 1e-5,
        min_gap: float = 1e-7,
        max_t: float = 1.0,
        prior_args: dict | None = None,
        sd_weight: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        torch.set_float32_matmul_precision("high")

    def forward(self, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        return self.net(x, s, t)

    @torch.no_grad()
    def sample_prior(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return self.manifold.prior(shape=shape, device=device)

    def sd_weight(self, s: Tensor, t: Tensor) -> Tensor | None:
        """
        Time-dependent weighting for self-distillation loss.
        """
        if self.hparams.sd_weight is None:
            return None
        elif self.hparams.sd_weight == "lin":
            return (t - s)
        raise ValueError(f"illegal choice of sd_weight {self.hparams.sd_weight}")

    def get_loss(self, x: Tensor | list[Tensor]) -> Tensor:
        if isinstance(x, list):
            x = x[0]

        with torch.no_grad():
            x0 = self.sample_prior(shape=x.shape, device=x.device)
            t = torch.rand(x.shape[0], device=x.device) * (1 - self.hparams.min_t) + self.hparams.min_t
            s = torch.rand(x.shape[0], device=x.device) * t
            s.clamp_(min=torch.tensor(0.0, device=self.device), max=t - self.hparams.min_gap)

            t = match_dims(t, x0.shape)
            s = match_dims(s, x0.shape)
        if self.hparams.underlying_loss == "esd":
            loss = self._esd_loss(x0, x, s, t)
        elif self.hparams.underlying_loss == "lsd":
            loss = self._lsd_loss(x0, x, s, t)
        elif self.hparams.underlying_loss == "psd":
            loss = self._psd_loss(x0, x, s, t)
        else:
            raise ValueError(f"illegal choice of loss {self.hparams.underlying_loss}")

        with torch.no_grad():
            sd_weight = self.sd_weight(s, t)
            if sd_weight is None:
                sd_weight = 1.0
            else:
                sd_weight = sd_weight.view(-1, *[1] * (len(x.shape) - 1))
        loss = (sd_weight * loss).mean()

        if abs(self.hparams.fm_loss_weight) > 1e-8:
            # calculate FM loss
            with torch.no_grad():
                xt, vt = self.manifold.geodesic_with_tangent(x0, x, t)
            if vt.isnan().any():
                nans = vt.isnan()
                while nans.ndim > 1:
                    nans = nans.any(dim=-1)
                vt = vt[~nans]
                xt = xt[~nans]
                t = t[~nans]
                self.log(
                    f"{self.get_stage()}/fm-nans",
                    nans.count_nonzero().detach().item(),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )
            out = self(xt, t, t)
            diff = out - vt
            # compute vf norm vs fm norm
            self.log(f"{self.get_stage()}/fm-norm", vt.detach().norm(dim=-1).mean(), on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"{self.get_stage()}/vf-norm", out.detach().norm(dim=-1).mean(), on_step=True, on_epoch=False, prog_bar=True)
            # cosine similarity
            with torch.no_grad():
                cos_sim = F.cosine_similarity(vt, out, dim=-1).abs().mean()
            self.log(f"{self.get_stage()}/cos-sim", cos_sim, on_step=True, on_epoch=False, prog_bar=True)
            fm_loss = self.manifold.inner(xt, diff, diff).mean()
            self.log(
                f"{self.get_stage()}/fm-loss",
                fm_loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                f"{self.get_stage()}/sd-loss",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            loss = loss + self.hparams.fm_loss_weight * fm_loss
        return loss if torch.isfinite(loss) else None

    def _esd_loss(self, x0: Tensor, x1: Tensor, s: Tensor, t: Tensor) -> Tensor:
        """
        Eulerian self-distillation loss.
        """
        with torch.no_grad():
            i_s = self._interpolant(x0, x1, s)
        # vs is partial_s of xst
        xst, vs = torch.func.jvp(
            lambda _s: self._xst(i_s, _s, t),
            primals=(s,),
            tangents=(torch.ones_like(s),),
        )
        # now, the pushforward
        with torch.no_grad():
            vss = self(i_s, s, s)
            _, push = torch.func.jvp(
                lambda z: self._xst(z, s, t),
                primals=(i_s,),
                tangents=(vss,),
            )
            # fix numerical precision
            push = self.manifold.project_tangent(i_s, vss)
        objective = vs + push
        loss = self.manifold.inner(xst, objective, objective)
        return loss

    def _lsd_loss(self, x0: Tensor, x1: Tensor, s: Tensor, t: Tensor) -> Tensor:
        """
        Lagrangian self-distillation loss.
        """
        with torch.no_grad():
            i_s = self._interpolant(x0, x1, s)
        xst, dvdt = torch.func.jvp(
            lambda _t: self._xst(i_s, s, _t),
            primals=(t,),
            tangents=(torch.ones_like(t),),
        )
        xst = xst.detach()
        with torch.no_grad():
            ref = self(xst, t, t)
        # assert self.manifold.all_belong_tangent(xst, ref)
        # assert self.manifold.all_belong_tangent(xst, dvdt)
        diff = ref - dvdt
        return self.manifold.inner(xst, diff, diff)

    def _psd_loss(self, x0: Tensor, x1: Tensor, s: Tensor, t: Tensor) -> Tensor:
        """
        Progressive self-distillation loss.
        """
        with torch.no_grad():
            u = 0.5 * s + 0.5 * t
            u.clamp_(min=s, max=t)
            i_s = self._interpolant(x0, x1, s)
            step_a = self._xst(i_s, s, u)
            step_b = self._xst(step_a, u, t)
        straight = self._xst(i_s, s, t)
        return self.manifold.distance(straight, step_b).square()

    def _xst(self, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        """
        Computes the xst through reparamterisation.
        """
        return self.manifold.exp(
            x,
            (t - s) * self(x, s, t),
        )

    def _beta(self, t: Tensor) -> Tensor:
        return t

    def _interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return self.manifold.geodesic_interpolant(x0, x1, self._beta(t))

    @torch.inference_mode()
    def sample_batch(self, steps: int, sz: int) -> Tensor:
        #returns x1
        timesteps = torch.linspace(0, 1, steps + 1, device=self.device)
        x = self.sample_prior(
            shape=(sz,) + tuple(self.hparams.in_shape), device=self.device
        )
        for s, t in zip(timesteps[:-1], timesteps[1:]):
            s = match_dims(s.expand(x.shape[0]), x.shape)
            t = match_dims(t.expand(x.shape[0]), x.shape)
            x = self.manifold.exp(
                x,
                (t - s) * self(x, s, t),
            )
        return x

    @property
    def is_sampling_time_zero_to_one(self) -> bool:
        return True

    @property
    def val_nll(self) -> bool:
        return self.hparams.eval_nll_exact

    def name(self) -> str:
        return f"flowmap-{self.hparams.underlying_loss}"
    
    #changed
    def load_VAE(self, VAE):
        self.VAE = VAE
        self.dataset_infos =  self.VAE.dataset_infos
    
    #changed
    def sample_decode(self, n_samples, x0=None, stage="valid"):
        x1, node_mask = self.sample(n_samples, x0)
        graph_list, graph_list_argmax = self.VE.generate_sample_from_z(x1, node_mask)
                
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
                _, _, hyp_feat_mean, hyp_feat_logvar, z = self.VAE.model.encode(x, adj, E, node_mask)
                # pdb.set_trace()
                node_feat=None
                edge_feat=None
                hyp_qz_x = WrappedNormalPoincare(hyp_feat_mean, F.softplus(hyp_feat_logvar), self.model.manifold)
                edge_feat = hyp_qz_x.rsample()
            
        return z, node_mask, node_feat, edge_feat
    