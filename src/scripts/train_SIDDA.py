import argparse
import math
import os
import random
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax import nnx

from augment import get_transform
from checkpointing import save_checkpoint
from dataset import NumpyLoader, dataset_dict, split_dataset
from models import model_dict
from sidda_losses import dynamic_sinkhorn_divergence, jensen_shannon_distance


def set_all_seeds(num: int) -> None:
    random.seed(num)
    np.random.seed(num)


class EtaParams(nnx.Module):
    """Learnable log-variance loss-weighting parameters (Kendall et al. 2018 style),
    trained jointly with the model via a separate optimizer sharing the model's
    lr-schedule/weight_decay (see train_SIDDA in the module docstring below)."""

    def __init__(self):
        self.eta_1 = nnx.Param(jnp.array(1.0))
        self.eta_2 = nnx.Param(jnp.array(1.0))


def build_lr_schedule(config, steps_per_epoch: int):
    milestones = config["parameters"]["milestones"]
    lr_decay = config["parameters"]["lr_decay"]
    boundaries_and_scales = {m * steps_per_epoch: lr_decay for m in milestones}
    return optax.piecewise_constant_schedule(
        init_value=config["parameters"]["lr"], boundaries_and_scales=boundaries_and_scales
    )


@nnx.jit
def train_step_ce(model, model_optimizer, x, y):
    """Warmup-phase step: source cross-entropy only, identical to train_CE.py."""

    def loss_fn(model):
        _, logits = model(x, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    model_optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step_ce(model, x, y):
    _, logits = model(x, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    predicted = logits.argmax(axis=-1)
    correct = (predicted == y).sum()
    return loss, correct


@nnx.jit
def train_step_sidda(model, eta_params, model_optimizer, eta_optimizer, source_x, source_y, target_x):
    """Post-warmup step: source CE + dynamic-blur Sinkhorn DA loss, combined via
    learnable log-variance weighting (eta_1, eta_2), gradient clipping applied to
    model params only (matching the original: clip_grad_norm_ was called with only
    model.parameters(), eta_1/eta_2's gradients were never clipped)."""

    def loss_fn(model, eta_params):
        batch_size = source_x.shape[0]
        concatenated = jnp.concatenate([source_x, target_x], axis=0)
        features, logits = model(concatenated, train=True)
        source_features = features[:batch_size]
        target_features = features[batch_size:]
        source_logits = logits[:batch_size]

        classification_loss = optax.softmax_cross_entropy_with_integer_labels(
            source_logits, source_y
        ).mean()

        da_loss, blur, max_distance = dynamic_sinkhorn_divergence(source_features, target_features)
        js_distance = jnp.nanmean(jensen_shannon_distance(source_features, target_features))

        eta_1 = eta_params.eta_1[...]
        eta_2 = eta_params.eta_2[...]
        loss = (
            (1 / (2 * eta_1**2)) * classification_loss
            + (1 / (2 * eta_2**2)) * da_loss
            + jnp.log(jnp.abs(eta_1) * jnp.abs(eta_2))
        )
        return loss, (classification_loss, da_loss, blur, max_distance, js_distance)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
    (loss, aux), (model_grads, eta_grads) = grad_fn(model, eta_params)

    # Clamp BEFORE this iteration's optimizer step, matching the original's exact
    # ordering (loss.backward(); ...; eta_1.data.clamp_(...); eta_2.data.clamp_(...);
    # optimizer.step()) -- the clamp there is applied to the *incoming* eta values
    # (computed by the previous iteration's raw, unclamped Adam update), not to the
    # freshly-updated ones. The fresh post-step result is therefore left unclamped
    # until the *next* iteration's clamp call. This is a looser, lagging enforcement
    # than clamping immediately after every update: clamping post-update (as an earlier
    # version of this port did) pins eta_2 to its floor every single step and prevents
    # its Adam trajectory from ever settling away from the constraint, which measurably
    # changes the DA-loss/classification-loss weighting balance over training.
    eta_params.eta_1[...] = jnp.clip(eta_params.eta_1[...], min=1e-3)
    eta_params.eta_2[...] = jnp.clip(eta_params.eta_2[...], min=0.25 * eta_params.eta_1[...])

    model_optimizer.update(model, model_grads)
    eta_optimizer.update(eta_params, eta_grads)

    return loss, aux


@nnx.jit
def eval_step_sidda(model, source_x, source_y, target_x):
    """Post-warmup validation step. NOTE: combined_loss here is the plain (unweighted)
    sum classification_loss + DA_loss, NOT the eta-weighted training loss -- this
    matches the original code's validation loop exactly."""
    batch_size = source_x.shape[0]
    concatenated = jnp.concatenate([source_x, target_x], axis=0)
    features, logits = model(concatenated, train=False)
    source_features = features[:batch_size]
    target_features = features[batch_size:]
    source_logits = logits[:batch_size]

    classification_loss = optax.softmax_cross_entropy_with_integer_labels(
        source_logits, source_y
    ).mean()
    da_loss, blur, max_distance = dynamic_sinkhorn_divergence(source_features, target_features)
    combined_loss = classification_loss + da_loss

    predicted = source_logits.argmax(axis=-1)
    correct = (predicted == source_y).sum()
    return combined_loss, classification_loss, da_loss, correct


def train_SIDDA(
    model,
    eta_params,
    train_loader: NumpyLoader,
    val_loader: NumpyLoader,
    target_loader: NumpyLoader,
    target_val_loader: NumpyLoader,
    model_optimizer,
    eta_optimizer,
    model_name: str,
    warmup: int,
    epochs: int = 100,
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    report_interval: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)

    print("Training Started!")
    best_val_acc = 0.0
    best_classification_loss = float("inf")
    best_DA_loss = float("inf")
    best_total_val_loss = float("inf")
    best_val_epoch = best_classification_loss_epoch = best_DA_epoch = 0
    no_improvement_count = 0

    losses, steps = [], []
    train_classification_losses, train_DA_losses = [], []
    val_losses, val_classification_losses, val_DA_losses = [], [], []
    max_distances, epoch_max_distances = [], []
    js_distances, epoch_js_distances = [], []
    blur_vals, epoch_blur_vals = [], []
    eta_1_vals, eta_2_vals = [], []

    for epoch in range(epochs):
        classification_losses, DA_losses = [], []
        train_loss = 0.0

        if epoch < warmup:
            for i, (source_batch, _target_batch) in enumerate(zip(train_loader, target_loader)):
                source_x, source_y = source_batch
                loss = train_step_ce(model, model_optimizer, source_x, source_y)
                loss = float(loss)
                train_loss += loss
                classification_losses.append(loss)
        else:
            for i, (source_batch, target_batch) in enumerate(zip(train_loader, target_loader)):
                source_x, source_y = source_batch
                target_x = target_batch[0] if isinstance(target_batch, tuple) else target_batch

                loss, aux = train_step_sidda(
                    model, eta_params, model_optimizer, eta_optimizer, source_x, source_y, target_x
                )
                classification_loss, da_loss, blur, max_distance, js_distance = aux

                train_loss += float(loss)
                classification_losses.append(float(classification_loss))
                DA_losses.append(float(da_loss))
                max_distances.append(float(max_distance))
                blur_vals.append(float(blur))
                js_distances.append(float(js_distance))
                eta_1_vals.append(float(eta_params.eta_1[...]))
                eta_2_vals.append(float(eta_params.eta_2[...]))

        n_train_batches = len(train_loader)
        train_loss /= n_train_batches
        train_classification_loss = float(np.mean(classification_losses))
        train_DA_loss = float(np.mean(DA_losses)) if DA_losses else None

        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_DA_losses.append(train_DA_loss)
        steps.append(epoch + 1)

        if epoch >= warmup and max_distances:
            mean_max_distance = float(np.mean(max_distances[-n_train_batches:]))
            epoch_max_distances.append(mean_max_distance)
            mean_blur_val = float(np.mean(blur_vals[-n_train_batches:]))
            epoch_blur_vals.append(mean_blur_val)
            mean_js_distance = float(np.nanmean(js_distances[-n_train_batches:]))
            epoch_js_distances.append(mean_js_distance)
            print(
                f"Epoch: {epoch + 1}, eta_1: {eta_params.eta_1[...]:.4f}, eta_2: {eta_params.eta_2[...]:.4f}"
            )
            print(f"Epoch: {epoch + 1}, Max Distance: {mean_max_distance:.4f}")
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}, DA Loss: {train_DA_loss:.4e}"
            )
        else:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}"
            )

        if (epoch + 1) % report_interval == 0:
            source_correct, source_total, val_loss = 0, 0, 0.0
            val_classification_loss, val_DA_loss = 0.0, 0.0

            for source_batch, target_batch in zip(val_loader, target_val_loader):
                source_x, source_y = source_batch
                target_x = target_batch[0] if isinstance(target_batch, tuple) else target_batch

                if epoch < warmup:
                    loss, correct = eval_step_ce(model, source_x, source_y)
                    val_loss += float(loss)
                    val_classification_loss += float(loss)
                else:
                    combined_loss, classification_loss_, da_loss_, correct = eval_step_sidda(
                        model, source_x, source_y, target_x
                    )
                    val_loss += float(combined_loss)
                    val_classification_loss += float(classification_loss_)
                    val_DA_loss += float(da_loss_)

                source_total += len(source_y)
                source_correct += int(correct)

            source_val_acc = 100 * source_correct / source_total
            n_val_batches = len(val_loader)
            val_loss /= n_val_batches
            val_classification_loss /= n_val_batches
            if epoch >= warmup:
                val_DA_loss /= n_val_batches

            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_DA_losses.append(val_DA_loss)

            if epoch < warmup:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}"
                )
            else:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}, Validation DA Loss: {val_DA_loss:.4e}"
                )

            # --- Best-checkpoint tracking: transcribed exactly from the original ---
            if val_loss < best_total_val_loss and epoch >= warmup:
                best_total_val_loss = val_loss
                best_val_epoch = epoch + 1
                save_checkpoint(model, os.path.join(save_dir, "best_model_total_val_loss"))
                print(f"Saved best total validation loss model at epoch {best_val_epoch}")
            else:
                no_improvement_count += 1

            if source_val_acc >= best_val_acc:
                best_val_acc = source_val_acc
                best_val_acc_epoch = epoch + 1
                save_checkpoint(model, os.path.join(save_dir, "best_model_val_acc"))
                print(f"Saved best validation accuracy model at epoch {best_val_acc_epoch}")

            if val_classification_loss <= best_classification_loss and epoch >= warmup:
                best_classification_loss = val_classification_loss
                best_classification_loss_epoch = epoch + 1
                save_checkpoint(model, os.path.join(save_dir, "best_model_classification_loss"))
                print(
                    f"Saved lowest classification loss model at epoch {best_classification_loss_epoch}"
                )

            if val_DA_loss <= best_DA_loss and epoch >= warmup:
                best_DA_loss = val_DA_loss
                best_DA_epoch = epoch + 1
                save_checkpoint(model, os.path.join(save_dir, "best_model_DA_loss"))
                print(f"Saved lowest DA loss model at epoch {best_DA_epoch}")

            # NOTE: no_improvement_count is only ever incremented above, never reset to
            # 0 on improvement -- a pre-existing bug carried over unfixed from the
            # original PyTorch code (see CLAUDE.md). Left as-is; sanity-check runs set
            # early_stopping_patience >= total epochs so this never binds.
            if no_improvement_count >= early_stopping_patience:
                print(
                    f"Early stopping after {early_stopping_patience} epochs without improvement in accuracy."
                )
                break

    save_checkpoint(model, os.path.join(save_dir, "final_model"))

    loss_dir = os.path.join(save_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    def _save(name, arr):
        # train_DA_losses mixes None (warmup epochs) with floats (post-warmup), which
        # numpy naturally infers as dtype=object; every other array is plain numeric.
        np.save(os.path.join(loss_dir, f"{name}-{model_name}.npy"), np.array(arr))

    _save("losses", losses)
    _save("train_classification_losses", train_classification_losses)
    _save("train_DA_losses", train_DA_losses)
    _save("val_losses", val_losses)
    _save("val_classification_losses", val_classification_losses)
    _save("val_DA_losses", val_DA_losses)
    _save("steps", steps)
    _save("max_distances", max_distances)
    _save("blur_vals", blur_vals)
    _save("js_distances", js_distances)
    _save("epoch_max_distances", epoch_max_distances)
    _save("epoch_blur_vals", epoch_blur_vals)
    _save("epoch_js_distances", epoch_js_distances)
    _save("eta_1_vals", eta_1_vals)
    _save("eta_2_vals", eta_2_vals)

    # --- Plotting (unchanged from the original: matplotlib is framework-agnostic) ---
    steps_arr = np.array(steps)
    validation_steps = steps_arr[::report_interval]
    losses_arr = np.array(losses, dtype=float)
    train_classification_losses_arr = np.array(train_classification_losses, dtype=float)
    train_DA_losses_arr = np.array(
        [v if v is not None else np.nan for v in train_DA_losses], dtype=float
    )
    val_losses_arr = np.array(val_losses, dtype=float)
    val_classification_losses_arr = np.array(val_classification_losses, dtype=float)
    val_DA_losses_arr = np.array(val_DA_losses, dtype=float)

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(steps_arr, losses_arr, label="Train Total Loss")
    plt.plot(steps_arr, train_classification_losses_arr, label="Train Classification Loss")
    plt.plot(steps_arr, train_DA_losses_arr, label="Train DA Loss")
    if best_val_epoch:
        plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    if best_classification_loss_epoch:
        plt.axvline(
            x=best_classification_loss_epoch, color="y", linestyle="--", label="Best Classification Epoch"
        )
    if best_DA_epoch:
        plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.yscale("log")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(validation_steps, val_losses_arr, label="Validation Total Loss")
    plt.plot(validation_steps, val_classification_losses_arr, label="Validation Classification Loss")
    plt.plot(validation_steps, val_DA_losses_arr, label="Validation DA Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"losses_plot-{model_name}.png"))
    plt.close()

    if epoch_max_distances:
        plt.figure(figsize=(10, 5))
        plt.plot(steps_arr[-len(epoch_max_distances):], epoch_max_distances)
        plt.xlabel("Epochs")
        plt.ylabel("Max Distance")
        plt.title("Max Distance vs. Training Steps")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir, f"max_distance_plot-{model_name}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(steps_arr[-len(epoch_blur_vals):], epoch_blur_vals)
        plt.axhline(y=0.01, color="r", linestyle="--")
        plt.axhline(y=0.05, color="g", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Blur Value")
        plt.title("Blur Value vs. Training Steps")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir, f"blur_value_plot-{model_name}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(steps_arr[-len(epoch_js_distances):], epoch_js_distances)
        plt.xlabel("Epochs")
        plt.ylabel("JS Distance")
        plt.title("JS Distance vs. Training Steps")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir, f"js_distance_plot-{model_name}.png"))
        plt.close()

    return (
        best_val_epoch,
        best_val_acc,
        best_classification_loss_epoch,
        best_classification_loss,
        best_DA_epoch,
        best_DA_loss,
        losses[-1],
    )


def main(config):
    model_name = str(config["model"]).strip()
    dataset_name = str(config["dataset"]).strip()

    rngs = nnx.Rngs(config["seed"])
    model = model_dict[dataset_name][model_name](rngs)
    eta_params = EtaParams()

    print("Loading datasets!")
    start = time.time()

    aug_rng = np.random.default_rng(config["seed"])
    train_transform = get_transform(dataset_name, train=True, rng=aug_rng)
    val_transform = get_transform(dataset_name, train=False, rng=aug_rng)

    source_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["input_path"],
        output_path=config["train_data"]["output_path"],
    )
    train_subset, val_subset = split_dataset(
        source_dataset,
        val_size=config["parameters"]["val_size"],
        seed=config["seed"],
        train_transform=train_transform,
        val_transform=val_transform,
    )

    target_output_path = config["train_data"].get("target_output_path")
    target_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["target_input_path"],
        output_path=target_output_path,
        target_domain=target_output_path is None,
    )
    target_subset, target_val_subset = split_dataset(
        target_dataset,
        val_size=config["parameters"]["val_size"],
        seed=config["seed"] + 1,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    end = time.time()
    print(f"Datasets loaded and split in {end - start} seconds")

    batch_size = config["parameters"]["batch_size"]
    # drop_last=True on all four: source and target domains can have different dataset
    # sizes (e.g. mrssc2's 4924 optical vs 4809 SAR images), so without it their final
    # batch of an epoch can differ in size. zip(train_loader, target_loader) pairs those
    # batches positionally regardless of size, and jensen_shannon_distance(source_features,
    # target_features) then fails to broadcast p + q whenever the paired batch sizes differ.
    train_loader = NumpyLoader(
        train_subset, batch_size=batch_size, shuffle=True, seed=config["seed"], drop_last=True
    )
    val_loader = NumpyLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    target_loader = NumpyLoader(
        target_subset, batch_size=batch_size, shuffle=True, seed=config["seed"] + 1, drop_last=True
    )
    target_val_loader = NumpyLoader(
        target_val_subset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    steps_per_epoch = len(train_subset) // batch_size
    lr_schedule = build_lr_schedule(config, steps_per_epoch)
    wd = config["parameters"]["weight_decay"]
    model_tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adamw(lr_schedule, weight_decay=wd))
    eta_tx = optax.adamw(lr_schedule, weight_decay=wd)

    model_optimizer = nnx.Optimizer(model, model_tx, wrt=nnx.Param)
    eta_optimizer = nnx.Optimizer(eta_params, eta_tx, wrt=nnx.Param)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config["save_dir"] + config["model"] + "_DA_" + timestr

    (
        best_val_epoch,
        best_val_acc,
        best_classification_epoch,
        best_classification_loss,
        best_DA_epoch,
        best_DA_loss,
        final_loss,
    ) = train_SIDDA(
        model=model,
        eta_params=eta_params,
        train_loader=train_loader,
        val_loader=val_loader,
        target_loader=target_loader,
        target_val_loader=target_val_loader,
        model_optimizer=model_optimizer,
        eta_optimizer=eta_optimizer,
        model_name=model_name,
        warmup=config["parameters"]["warmup"],
        epochs=config["parameters"]["epochs"],
        save_dir=save_dir,
        early_stopping_patience=config["parameters"]["early_stopping"],
        report_interval=config["parameters"]["report_interval"],
    )
    print("Training Done")
    config["best_val_acc"] = best_val_acc
    config["best_val_epoch"] = best_val_epoch
    config["final_loss"] = float(final_loss)
    config["best_classification_epoch"] = best_classification_epoch
    config["best_classification_loss"] = best_classification_loss
    config["best_DA_epoch"] = best_DA_epoch
    config["best_DA_loss"] = best_DA_loss

    with open(f"{save_dir}/config.yaml", "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the models")
    parser.add_argument(
        "--config", metavar="config", required=True, help="Location of the config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config["seed"])

    main(config)
