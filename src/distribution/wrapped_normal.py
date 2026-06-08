import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all
import pdb

def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k**2 * x2 * y2

    # Example usage
    print("Checking num:")
    # print({type(num)})
    print({num.shape})
    print({num.min().item()})
    print({num.max().item()})
    print({torch.isnan(num).any().item()})
    print({torch.isinf(num).any().item()})
    print()
    
    print("Checking denom:")
    # print({type(denom)})
    print({denom.shape})
    print({denom.min().item()})
    print({denom.max().item()})
    print({torch.isnan(denom).any().item()})
    print({torch.isinf(denom).any().item()})
    print()

    return num / denom.clamp_min(1e-15)

def _lambda_x(x: torch.Tensor, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return 2 / (1 + k * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(1e-15)

def sign(x):
    return torch.sign(x.sign() + 0.5)

def abs_zero_grad(x):
    return x * sign(x)


def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
            - 1 / 11 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x**3
    elif order == 2:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5
    elif order == 3:
        return (
            x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5 - 1 / 7 * k**3 * x**7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)

def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


def _logmap(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    sub = _mobius_add(-x, y, k, dim=dim)
    sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    lam = _lambda_x(x, k, keepdim=True, dim=dim)
    return 2.0 * artan_k(sub_norm, k) * (sub / (lam * sub_norm))



class WrappedNormal(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.device = loc.device
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold.to(self.device)
        self.manifold.assert_check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        pdb.set_trace()
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)
        v = v / self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        u = self.manifold.transp(self.manifold.zero, self.loc, v)
        z = self.manifold.expmap(self.loc, u)
        return z

    def log_prob(self, x):
        shape = x.shape
        loc = self.loc.unsqueeze(0).expand(x.shape[0], *self.batch_shape, self.manifold.coord_dim)
        if len(shape) < len(loc.shape): x = x.unsqueeze(1)
        v = self.manifold.logmap(loc, x)
        v = self.manifold.transp(loc, self.manifold.zero, v)
        u = v * self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        norm_pdf = Normal(torch.zeros_like(self.scale), self.scale).log_prob(u).sum(-1, keepdim=True)
        logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)
        result = norm_pdf - logdetexp
        return result
     


class WrappedNormalPoincare(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        # return F.softplus(self._scale) if self.softplus else self._scale
        return self._scale
    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.device = loc.device
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold.to(self.device)
        self.manifold.assert_check_point_on_manifold(self.loc)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.loc.shape[-1]])
        super(WrappedNormalPoincare, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        d = self.loc.size(-1) 
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device) 
        # / d ** 0.5
        
        self.manifold.assert_check_vector_on_tangent(self.manifold.zero(self.loc.shape[-1]), v)
        # self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)
        # Ensure lambda_x calculation is stable
        lambda_factor = self.manifold.lambda_x(self.manifold.zero(self.loc.shape[-1]), keepdim=True)
        # lambda_factor = self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        v = v / lambda_factor
        
        # Transport vector from tangent space at origin to tangent space at loc
        u = self.manifold.transp(self.manifold.zero(self.loc.shape[-1]), self.loc, v)
        # u = self.manifold.transp(self.manifold.zero, self.loc, v)
        # Map the tangent vector to the manifold
        z = self.manifold.expmap(self.loc, u)
        
        # Final projection to ensure point is on manifold
        if hasattr(self.manifold, 'proj'):
            z = self.manifold.proj(z)  
        return z


    def log_prob(self, x):
        try:
            shape = x.shape
            loc = self.loc
            if len(shape) < len(loc.shape): x = x.unsqueeze(1)
            
            # Ensure points are on the manifold before computing logmap
            if hasattr(self.manifold, 'proj'):
                x = self.manifold.proj(x)
            
            # Safely compute logmap with error handling
            try:
                v = self.manifold.logmap(loc, x)
            except Exception as e:
                print(f"Error in logmap: {e}")
                # Return a reasonable default log probability
                return torch.ones_like(x[..., 0:1]) * -100
            
            # Check for problematic values
            if torch.isnan(v).any() or torch.isinf(v).any():
                print("Warning: NaN or Inf detected in log_prob after logmap")
                # Replace NaN/Inf with zeros to avoid propagation
                v = torch.where(torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v)
            
            # Transport vector safely
            v = self.manifold.transp(loc, self.manifold.zero(self.loc.shape[-1]), v)
            
            # Safely compute lambda_x
            lambda_factor = self.manifold.lambda_x(self.manifold.zero(self.loc.shape[-1]), keepdim=True)
            if torch.isnan(lambda_factor).any() or torch.isinf(lambda_factor).any():
                print("Warning: NaN or Inf detected in lambda_factor in log_prob")
                lambda_factor = torch.where(
                    torch.isnan(lambda_factor) | torch.isinf(lambda_factor),
                    torch.ones_like(lambda_factor),
                    lambda_factor
                )
            
            u = v * lambda_factor
            
            # Safely compute log probability in normal distribution
            d = u.size(-1)
            sigma_prime = self.scale 
            # / d ** 0.5
            norm_pdf = Normal(torch.zeros_like(sigma_prime), sigma_prime.clamp_min(1e-5)).log_prob(u).sum(-1, keepdim=True)
            
            # Safely compute logdetexp
            try:
                logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)
                # Clamp to prevent extreme values
                logdetexp = torch.clamp(logdetexp, min=-100, max=100)
            except Exception as e:
                print(f"Error in logdetexp: {e}")
                logdetexp = torch.zeros_like(norm_pdf)
            
            result = norm_pdf - logdetexp
            
            # Handle any remaining NaN or inf values in final result
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN or Inf in final log_prob result")
                result = torch.where(
                    torch.isnan(result) | torch.isinf(result), 
                    torch.ones_like(result) * -100,  # default to very low probability
                    result
                )
            
            return result
        
        except Exception as e:
            print(f"Error in WrappedNormalPoincare.log_prob: {e}")
            # Return a reasonable default log probability
            return torch.ones_like(x[..., 0:1]) * -100