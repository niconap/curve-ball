from __future__ import annotations

from abc import ABC, abstractmethod
from geoopt.manifolds.stereographic import math as stereo_math
import geoopt
from torch import Tensor
import torch


class Manifold(ABC):
    """Defines the manifold abstraction."""

    @abstractmethod
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Defines the exponential map at `p` in the direction `v`.
        """

    @abstractmethod
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Defines the logarithmic map from `p` to `q`.
        """

    @abstractmethod
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Returns the geodesic distance of points `p` and `q` on the manifold.
        """

    @abstractmethod
    def all_belong(self, p: Tensor) -> bool:
        """
        Returns `true` iff every point in `p` (batched) is in the manifold `self`.
        """

    @abstractmethod
    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        """
        Returns `true` iff every point in `p` (batched) and tangent vector `v`
        belong to the tangent space of the manifold at `p`.
        """

    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Projects the points `x` to the manifold.
        """

    @abstractmethod
    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Projects the tangent vector `v` at point `x` to the tangent space of the manifold.
        """

    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Returns the geodesic interpolant at time `t`, i.e.,
        `exp_{x_0}(t log_{x_0}(x_1))`.
        """
        # assert self.all_belong(x0)
        # assert self.all_belong(x1)
        lg = self.log(x0, x1)
        while len(t.shape) < len(lg.shape):
            t = t[..., None]
        return self.project(self.exp(x0, t * lg))

    def geodesic_with_tangent(
        self, x0: Tensor, x1: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns `xt` with its time derivative.
        """
        return torch.func.jvp(
            lambda _t: self.geodesic_interpolant(x0, x1, _t),
            primals=(t,),
            tangents=(torch.ones_like(t),),
        )

    @abstractmethod
    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        """
        Returns samples from the default prior on the given manifold.
        """

    @abstractmethod
    def logp0(self, x: Tensor) -> Tensor:
        """
        Returns the log-probability of the point `x` under the default prior on the manifold.
        """

    @abstractmethod
    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Returns the inner product a point `p` between `u` and `v`.
        """

    @abstractmethod
    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        Returns the parallely transported `v` from `p` to `q`.
        """
        
class PoincareBall(Manifold):
    # NOTE: k is explicited almost everywhere for JVP

    def __init__(self):
        self.geoopt = geoopt.manifolds.PoincareBall()

    def get_type(self):
        return super().get_type()
    
    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1):
        k = -1.0
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
        denom = 1 - 2 * k * xy + k**2 * x2 * y2
        # minimize denom (omit K to simplify th notation)
        # 1)
        # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
        # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
        # 2)
        # {y + x * <y, y> = 0
        # {x + y * <x, x> = 0
        # 3)
        # {- y/<y, y> = x
        # {- x/<x, x> = y
        # 4)
        # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
        return num / denom.clamp_min(1e-15)

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        x = p
        u = v
        dim = -1
        # copy content to avoid JVP issues
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
        lam = self._lambda_x(x)
        second_term = self.tan_k((lam / 2.0) * u_norm) * (u / u_norm)
        y = self._mobius_add(x, second_term, dim=dim)
        return y
        # return GeooptPoincareBall().expmap(p, v)

    def artanh(self, x: Tensor):
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

    def artan_k(self, x: Tensor):
        return self.artanh(x)

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        x = p
        y = q
        dim = -1
        # return self.geoopt.logmap(p, q)
        # return stereo_math.logmap(p, q, k=torch.tensor(-1.0, device=p.device))
        sub = self._mobius_add(-x, y, dim=dim)
        sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
        lam = self._lambda_x(x)
        return 2.0 * self.artan_k(sub_norm) * (sub / (lam * sub_norm))
        # return GeooptPoincareBall().logmap(p, q)

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # return self.geoopt.inner(p, u, v)
        return stereo_math.inner(p, u, v, k=torch.tensor(-1.0, device=p.device))
        # return GeooptPoincareBall().inner(p, u, v)

    def prior(self, shape: tuple[int, ...], device: str | torch.device = "cpu") -> Tensor:
        distance = 0.6
        std = 0.7
        sign0 = torch.tensor(-1.0, device=device)
        mean0 = torch.tensor([distance, distance], device=device) * sign0

        v = torch.randn(shape, device=device) * std
        lambda_x = self._lambda_x(mean0).unsqueeze(-1)
        return self.exp(mean0, v / lambda_x)

    def logp0(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError()

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        return self.geoopt.proju(x, v)

    def all_belong(self, p: Tensor) -> bool:
        return self.geoopt.check_point_on_manifold(p)

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        return self.geoopt.check_vector_on_tangent(p, v)

    def tan_k(self, x: torch.Tensor):
        k_sqrt = torch.tensor(1.0, device=x.device)
        scaled_x = x * k_sqrt

        return k_sqrt.reciprocal() * scaled_x.clamp(-15, 15).tanh()

    def project(self, x: Tensor) -> Tensor:
        dim = -1
        # return self.geoopt.projx(x)
        # return stereo_math.project(x, k=torch.tensor(-1.0, device=x.device))
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
        maxnorm = (1 - eps)
        # maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
        norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
        # return GeooptPoincareBall().projx(x)

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        # return self.geoopt.dist(p, q)
        return geoopt.manifolds.PoincareBall().dist(p, q)

    def _lambda_x(self, x) -> Tensor:
        # lambda_x = 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)
        # for k = -1
        return 2 / (1 - x.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)

    def metric_normalize(self, x: Tensor, v: Tensor) -> Tensor:
        # return v / self.geoopt.lambda_x(x, keepdim=True)
        return v / self._lambda_x(x)
        # return v / GeooptPoincareBall().lambda_x(x, keepdim=True)
        # return v
    
    def expmap(self, x: Tensor, v: Tensor) -> Tensor:
        return self.exp(x, v)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return self.log(x, y)

    def dist(self, x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
        d = self.distance(x, y)
        return d.unsqueeze(-1) if keepdim else d

    def proju(self, x: Tensor, v: Tensor) -> Tensor:
        return self.project_tangent(x, v)

    def projx(self, x: Tensor) -> Tensor:
        return self.project(x)

    def expmap0(self, v: Tensor) -> Tensor:
        return self.exp(torch.zeros_like(v), v)

    def logmap0(self, x: Tensor) -> Tensor:
        return self.log(torch.zeros_like(x), x)
    
    def logdetexp(
        self,
        x: Tensor,
        v: Tensor,
        is_vector: bool = True,
        keepdim: bool = False,
    ) -> Tensor:
        dim = x.shape[-1]
        if is_vector:
            lam = self._lambda_x(x)
            r = (lam / 2.0) * v.norm(dim=-1, keepdim=keepdim, p=2)
        else:
            r = v
        r = r.clamp_min(1e-15)
        return (dim - 1) * (torch.sinh(r) / r).log()

    def random(self, *shape, std: float = 1.0, device=None, dtype=None) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        v = torch.randn(*shape, device=device, dtype=dtype) * std
        return self.expmap0(v)

    def random_base(
        self, n_samples: int, dim: int, std: float = 1.0, device=None, dtype=None
    ) -> Tensor:
        v = torch.randn(n_samples, dim, device=device, dtype=dtype) * std
        return self.expmap0(v)

    def base_logprob(self, x: Tensor) -> Tensor:
        import math as _math
        dim = x.shape[-1]
        v = self.logmap0(x)
        log_p_euc = (
            -0.5 * dim * _math.log(2 * _math.pi)
            - 0.5 * (v * v).sum(dim=-1)
        )
        log_det = self.logdetexp(torch.zeros_like(x), v, is_vector=True)
        return log_p_euc - log_det