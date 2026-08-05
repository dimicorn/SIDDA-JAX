import argparse
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax import nnx

from augment import get_transform
from checkpointing import save_checkpoint
from dataset import NumpyLoader, dataset_dict, split_dataset
from models import model_dict


def set_all_seeds(num: int) -> None:
    random.seed(num)
    np.random.seed(num)


def build_lr_schedule(config, steps_per_epoch: int):
    milestones = config["parameters"]["milestones"]
    lr_decay = config["parameters"]["lr_decay"]
    boundaries_and_scales = {m * steps_per_epoch: lr_decay for m in milestones}
    return optax.piecewise_constant_schedule(
        init_value=config["parameters"]["lr"], boundaries_and_scales=boundaries_and_scales
    )


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        _, logits = model(x, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    _, logits = model(x, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    predicted = logits.argmax(axis=-1)
    correct = (predicted == y).sum()
    return loss, correct


def train_model(
    model,
    train_loader: NumpyLoader,
    val_loader: NumpyLoader,
    optimizer,
    model_name: str,
    epochs: int = 100,
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    report_interval: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)

    print("Training Started!")
    best_val_acc, no_improvement_count = 0.0, 0
    best_val_epoch = 0
    losses, steps = [], []

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            loss = train_step(model, optimizer, imgs, labels)
            loss = float(loss)
            train_loss += loss
            losses.append(loss)
            steps.append(epoch * len(train_loader) + i + 1)

        train_loss /= len(train_loader)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")

        if (epoch + 1) % report_interval == 0:
            correct, total, val_loss = 0, 0, 0.0
            for imgs, labels in val_loader:
                loss, batch_correct = eval_step(model, imgs, labels)
                val_loss += float(loss)
                correct += int(batch_correct)
                total += len(labels)

            val_acc = 100 * correct / total
            val_loss /= len(val_loader)
            print(
                f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                save_checkpoint(model, os.path.join(save_dir, "best_model_val_acc"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(
                    f"Early stopping after {early_stopping_patience} epochs without improvement."
                )
                break

    save_checkpoint(model, os.path.join(save_dir, "final_model"))

    loss_dir = save_dir
    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Training Steps")
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches="tight")
    plt.close()

    return best_val_epoch, best_val_acc, losses[-1]


def main(config):
    model_name = str(config["model"]).strip()
    dataset_name = str(config["dataset"]).strip()

    rngs = nnx.Rngs(config["seed"])
    model = model_dict[dataset_name][model_name](rngs)

    print("Loading datasets!")
    start = time.time()

    train_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["input_path"],
        output_path=config["train_data"]["output_path"],
    )

    aug_rng = np.random.default_rng(config["seed"])
    # Opt-IN per-dataset flag, default False. The original PyTorch code's
    # transform-aliasing bug (see CLAUDE.md) meant its published results were trained
    # with effectively zero augmentation despite the paper's stated recipe (+-180
    # degree rotation, flips, +-10% translation); replaying that recipe for real
    # measurably breaks reproduction -- mnist_m's digit labels aren't invariant to
    # +-180 degree rotation (a 6 rotated ~180 degrees looks like a 9), and even on
    # rotation-invariant-label datasets (shapes) the combination of this augmentation
    # strength with lr=1e-2 destabilizes optimization (see CLAUDE.md for measurements).
    use_augment = config["parameters"].get("augment", False)
    train_transform = get_transform(dataset_name, train=use_augment, rng=aug_rng)
    val_transform = get_transform(dataset_name, train=False, rng=aug_rng)

    train_subset, val_subset = split_dataset(
        train_dataset,
        val_size=config["parameters"]["val_size"],
        seed=config["seed"],
        train_transform=train_transform,
        val_transform=val_transform,
    )

    end = time.time()
    print(f"Datasets loaded and split in {end - start} seconds")

    batch_size = config["parameters"]["batch_size"]
    train_loader = NumpyLoader(train_subset, batch_size=batch_size, shuffle=True, seed=config["seed"])
    val_loader = NumpyLoader(val_subset, batch_size=batch_size, shuffle=False)

    steps_per_epoch = math.ceil(len(train_subset) / batch_size)
    lr_schedule = build_lr_schedule(config, steps_per_epoch)
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(lr_schedule, weight_decay=config["parameters"]["weight_decay"]),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config["save_dir"] + config["model"] + "_" + timestr

    best_val_epoch, best_val_acc, final_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        model_name=model_name,
        epochs=config["parameters"]["epochs"],
        save_dir=save_dir,
        early_stopping_patience=config["parameters"]["early_stopping"],
        report_interval=config["parameters"]["report_interval"],
    )
    print("Training Done")
    config["best_val_acc"] = best_val_acc
    config["best_val_epoch"] = best_val_epoch
    config["final_loss"] = float(final_loss)

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
