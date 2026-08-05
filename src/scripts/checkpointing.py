"""Shared Orbax-based checkpoint save/restore for the Flax NNX models in this pipeline.

Under Orbax every checkpoint is a directory, not a flat file (unlike the original
PyTorch code's `*.pt` files) -- discovery must therefore allowlist-filter by name, since
a naive "any subdirectory" scan would also try to restore a run's `losses/`, `metrics/`,
`confusion_matrix/`, `latent_vectors/`, and `y_pred/` output directories and crash.
"""

import os
from typing import List, Tuple

import orbax.checkpoint as ocp
from flax import nnx

from models import model_dict

KNOWN_CHECKPOINT_NAMES = {
    "best_model_val_acc",
    "best_model_total_val_loss",
    "best_model_classification_loss",
    "best_model_DA_loss",
    "final_model",
}


def save_checkpoint(model: nnx.Module, path: str) -> None:
    """Save full model state (both nnx.Param AND nnx.BatchStat -- BatchNorm running
    mean/var -- dropping BatchStats would silently break eval-mode inference after
    restore, the JAX analogue of PyTorch's state_dict() already including buffers).
    """
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(path), state, force=True)
    checkpointer.wait_until_finished()
    checkpointer.close()


def load_models(directory_path: str, model_name: str, dataset_name: str) -> List[Tuple[nnx.Module, str]]:
    """Load every known checkpoint kind present in a run directory.

    Returns:
        list of (model, checkpoint_name) tuples, analogous to the original PyTorch
        load_models' [(model, model_name_no_ext), ...] return shape.
    """
    models = []
    checkpointer = ocp.StandardCheckpointer()

    for entry in sorted(os.listdir(directory_path)):
        full_path = os.path.join(directory_path, entry)
        if entry not in KNOWN_CHECKPOINT_NAMES or not os.path.isdir(full_path):
            continue

        print(f"Loading {model_name} from {full_path}...")
        abstract_model = nnx.eval_shape(
            lambda: model_dict[dataset_name][model_name](rngs=nnx.Rngs(0))
        )
        graphdef, abstract_state = nnx.split(abstract_model)
        state = checkpointer.restore(os.path.abspath(full_path), abstract_state)
        model = nnx.merge(graphdef, state)
        models.append((model, entry))
        print(f"Finished loading {model_name} from {full_path}")

    checkpointer.close()

    if not models:
        print(
            f"No checkpoints matching {sorted(KNOWN_CHECKPOINT_NAMES)} found in {directory_path}."
        )

    return models
