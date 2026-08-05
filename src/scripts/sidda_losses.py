"""Domain-adaptation loss helpers for SIDDA: Jensen-Shannon diagnostic distance and
the dynamic-blur Sinkhorn divergence between source/target latent features.

The Sinkhorn divergence here is built to match ``geomloss.SamplesLoss("sinkhorn", p=2,
blur=b, scaling=0.9, reach=None)`` (debiased, default settings) as used by the original
PyTorch implementation:
  - geomloss's ground cost for p=2 is C(x, y) = 0.5 * ||x - y||^2, and its entropic
    regularization temperature is epsilon = blur ** 2. ott's built-in
    ``costs.SqEuclidean`` returns ||x - y||^2 with NO 1/2 factor, so it is paired here
    with ``HalfSqEuclidean`` to reproduce geomloss's Gibbs kernel exactly.
  - geomloss's ``scaling=0.9`` controls epsilon-annealing in its solver, a numerical
    acceleration/conditioning device with no effect on the converged fixed point and no
    direct equivalent in ott-jax's plain Sinkhorn solver. We use a fixed epsilon (no
    annealing) here -- an accepted, documented behavioral difference, not a blocker.
"""

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from ott.geometry import costs, pointcloud
from ott.tools import sinkhorn_divergence as ott_sinkhorn_divergence


@jtu.register_pytree_node_class
class HalfSqEuclidean(costs.SqEuclidean):
    """C(x, y) = 0.5 * ||x - y||^2 -- matches geomloss's p=2 cost convention.

    Must be re-registered as its own pytree node class (not inherited from
    SqEuclidean's registration) so JAX can pass it through jit/grad transforms.
    """

    def __call__(self, x, y):
        return 0.5 * super().__call__(x, y)


def kl_divergence(p, q):
    epsilon = 1e-6
    p = jnp.clip(p, min=epsilon)
    q = jnp.clip(q, min=epsilon)
    return jnp.sum(p * jnp.log(p / q), axis=-1)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = jnp.clip(jsd, min=0.0)
    return jnp.sqrt(jsd)


def pairwise_l2(x, y):
    """Unsquared Euclidean distance matrix, equivalent to torch.cdist(x, y, p=2)."""
    sq = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.sqrt(jnp.clip(sq, min=0.0))


def dynamic_blur(source_features, target_features, min_blur: float = 0.01):
    """Dynamic Sinkhorn blur value: 0.05 * max pairwise source/target distance,
    lower-bounded at ``min_blur``. Stop-gradient'd, matching the original's
    ``max_distance.detach()`` -- gradients flow through the Sinkhorn divergence's value
    at this blur, never through how the blur itself was chosen.
    """
    distances = pairwise_l2(source_features, target_features)
    max_distance = jax.lax.stop_gradient(jnp.max(distances))
    blur = jnp.maximum(0.05 * max_distance, min_blur)
    return blur, max_distance


def sinkhorn_loss(x, y, blur):
    """Debiased Sinkhorn divergence between two point clouds at the given blur,
    matching ``geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)``.
    """
    epsilon = blur**2
    divergence, _ = ott_sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=x,
        y=y,
        cost_fn=HalfSqEuclidean(),
        epsilon=epsilon,
    )
    return divergence


def dynamic_sinkhorn_divergence(source_features, target_features, min_blur: float = 0.01):
    """Full DA-loss step: dynamic blur derivation + Sinkhorn divergence at that blur.

    Returns (divergence, blur, max_distance).
    """
    blur, max_distance = dynamic_blur(source_features, target_features, min_blur=min_blur)
    divergence = sinkhorn_loss(source_features, target_features, blur)
    return divergence, blur, max_distance
