import jax
import jax.numpy as jnp
from flax import nnx

# PyTorch's nn.Conv2d default init (no explicit override, used for conv1/conv2/conv3 in
# both the original and here) is kaiming_uniform_(weight, a=sqrt(5)), which reduces to
# Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)); bias defaults to the same Uniform bound.
# Flax's nnx.Conv default kernel_init has a ~2x larger std and its default bias_init is
# zeros -- both need to be matched explicitly, since these convs have no explicit
# override in the original PyTorch code either (only fc1/fc2 do) and left as Flax
# defaults this measurably changes the trained latent feature scale/separation between
# domains, which cascades into the dynamic Sinkhorn blur and the whole DA-loss weighting
# balance during SIDDA training.
_PT_CONV_KERNEL_INIT = jax.nn.initializers.variance_scaling(
    scale=1 / 3, mode="fan_in", distribution="uniform"
)


def _pt_conv_bias_init(fan_in: int):
    bound = fan_in**-0.5

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class CNN(nnx.Module):
    """CNN model (Flax NNX). NHWC input; forward returns (latent_space, logits).

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 1.
        num_classes (int, optional): Number of classes. Defaults to 3.
        input_size (tuple, optional): (H, W) input size, used only to dynamically
            derive the flatten size feeding the bottleneck FC layer. Defaults to (100, 100).
        rngs (nnx.Rngs): RNG state for parameter init and dropout.
    """

    def __init__(
        self,
        num_channels: int = 1,
        num_classes: int = 3,
        input_size: tuple = (100, 100),
        *,
        rngs: nnx.Rngs,
    ):
        # NOTE: Flax's BatchNorm `momentum` is the *retention* weight on the OLD running
        # stat (running = momentum*old + (1-momentum)*new) -- the opposite convention
        # from PyTorch, where `momentum` is the weight on the NEW batch stat (running =
        # (1-momentum)*old + momentum*new). PyTorch's nn.BatchNorm2d default is
        # momentum=0.1 (10% weight on each new batch); to match that behavior here we
        # need momentum=1-0.1=0.9, NOT Flax's own default of 0.99 (which would only
        # give 1% weight per batch -- a 10x slower-adapting running average).
        bn_momentum = 0.9

        self.conv1 = nnx.Conv(
            num_channels,
            8,
            kernel_size=(5, 5),
            padding="SAME",
            kernel_init=_PT_CONV_KERNEL_INIT,
            bias_init=_pt_conv_bias_init(num_channels * 5 * 5),
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(8, momentum=bn_momentum, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.2, rngs=rngs)

        self.conv2 = nnx.Conv(
            8,
            16,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=_PT_CONV_KERNEL_INIT,
            bias_init=_pt_conv_bias_init(8 * 3 * 3),
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(16, momentum=bn_momentum, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.2, rngs=rngs)

        self.conv3 = nnx.Conv(
            16,
            32,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=_PT_CONV_KERNEL_INIT,
            bias_init=_pt_conv_bias_init(16 * 3 * 3),
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(32, momentum=bn_momentum, rngs=rngs)
        self.dropout3 = nnx.Dropout(0.2, rngs=rngs)

        # Dynamically compute the post-conv/pool flatten size via a real (eager, not
        # jitted) forward pass through the already-constructed conv/pool stack, mirroring
        # the original PyTorch code's dummy-forward approach. Three VALID stride-2
        # 2x2 maxpools on e.g. a 100x100 input give 100->50->25->12 (floor division at
        # the odd 25 step), not a naive /8 -- don't hand-derive this.
        dummy = jnp.zeros((1, input_size[0], input_size[1], num_channels))
        x = self._conv_pool_stack(dummy, train=False)
        flattened_size = x.reshape(1, -1).shape[1]

        self.fc1 = nnx.Linear(
            flattened_size,
            256,
            kernel_init=nnx.initializers.normal(0.005),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layer_norm = nnx.LayerNorm(256, rngs=rngs)
        self.fc2 = nnx.Linear(
            256,
            num_classes,
            kernel_init=nnx.initializers.normal(0.01),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def _conv_pool_stack(self, x, *, train: bool):
        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not train))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = self.dropout1(x, deterministic=not train)

        x = nnx.relu(self.bn2(self.conv2(x), use_running_average=not train))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = self.dropout2(x, deterministic=not train)

        x = nnx.relu(self.bn3(self.conv3(x), use_running_average=not train))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = self.dropout3(x, deterministic=not train)
        return x

    def __call__(self, x, *, train: bool = False):
        x = self._conv_pool_stack(x, train=train)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        latent_space = self.layer_norm(x)
        logits = self.fc2(latent_space)
        return latent_space, logits


##############################################################################################


def cnn_shapes(rngs: nnx.Rngs):
    return CNN(num_channels=1, num_classes=3, input_size=(100, 100), rngs=rngs)


def cnn_astro_objects(rngs: nnx.Rngs):
    return CNN(num_channels=1, num_classes=3, input_size=(100, 100), rngs=rngs)


def cnn_mnistm(rngs: nnx.Rngs):
    return CNN(num_channels=3, num_classes=10, input_size=(32, 32), rngs=rngs)


def cnn_gzevo(rngs: nnx.Rngs):
    return CNN(num_channels=3, num_classes=6, input_size=(100, 100), rngs=rngs)


def cnn_mrssc2(rngs: nnx.Rngs):
    return CNN(num_channels=3, num_classes=7, input_size=(100, 100), rngs=rngs)


shapes_models = {"cnn": cnn_shapes}
astro_objects_models = {"cnn": cnn_astro_objects}
mnistm_models = {"cnn": cnn_mnistm}
gz_evo_models = {"cnn": cnn_gzevo}
mrssc2_models = {"cnn": cnn_mrssc2}

model_dict = {
    "shapes": shapes_models,
    "astro_objects": astro_objects_models,
    "mnist_m": mnistm_models,
    "gz_evo": gz_evo_models,
    "mrssc2": mrssc2_models,
}


if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = CNN(num_channels=3, num_classes=10, input_size=(32, 32), rngs=rngs)
    dummy = jnp.zeros((4, 32, 32, 3))
    latent, logits = model(dummy, train=False)
    print("latent shape:", latent.shape)
    print("logits shape:", logits.shape)
