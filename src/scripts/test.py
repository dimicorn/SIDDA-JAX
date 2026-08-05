import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from augment import get_transform
from checkpointing import load_models
from dataset import NumpyLoader, classes_dict, dataset_dict


def compute_metrics(
    test_loader: NumpyLoader,
    model,
    model_name: str,
    save_dir: str,
    output_name: str,
    classes: tuple,
):
    """Compute metrics for the model.

    Returns:
        sklearn_report (dict): sklearn classification report
    """
    y_pred, y_true, feature_maps = [], [], []

    for imgs, labels in test_loader:
        features, logits = model(imgs, train=False)
        predicted_class = logits.argmax(axis=-1)
        feature_maps.extend(np.asarray(features))
        y_pred.extend(np.asarray(predicted_class))
        y_true.extend(np.asarray(labels))

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)

    features_dir = os.path.join(save_dir, "latent_vectors")
    os.makedirs(features_dir, exist_ok=True)
    y_pred_dir = os.path.join(save_dir, "y_pred")
    os.makedirs(y_pred_dir, exist_ok=True)
    np.save(
        f"{features_dir}/latent_vecs_{model_name}_{output_name}.npy", flattened_features
    )
    np.save(f"{y_pred_dir}/y_pred_{model_name}_{output_name}.npy", y_pred)

    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrix")
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    sklearn_report = classification_report(
        y_true, y_pred, output_dict=True, target_names=classes, labels=range(len(classes))
    )

    cf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    df_cm = pd.DataFrame(
        cf_matrix / np.maximum(np.sum(cf_matrix, axis=1)[:, None], 1),
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(
        os.path.join(
            confusion_matrix_dir, f"confusion_matrix_{model_name}_{output_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()

    return sklearn_report


def main(
    model_dir: str,
    output_name: str,
    x_test_path: str,
    y_test_path: str,
    model_name: str,
    classes: tuple,
    dataset: str,
):
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    eval_rng = np.random.default_rng(0)
    transform = get_transform(dataset, train=False, rng=eval_rng)

    test_dataset = dataset_dict[dataset](x_test_path, y_test_path, transform=transform)
    test_dataloader = NumpyLoader(test_dataset, batch_size=128, shuffle=False)

    models = load_models(model_dir, model_name, dataset)
    if not models:
        print("Models could not be loaded.")
        return

    for model, checkpoint_name in models:
        full_report = compute_metrics(
            test_loader=test_dataloader,
            model=model,
            model_name=model_name,
            save_dir=model_dir,
            output_name=f"{output_name}_{checkpoint_name}",
            classes=classes,
        )

        print("Compiling Metrics")
        output_file_name = f"{output_name}_{checkpoint_name}.yaml"
        with open(os.path.join(metrics_dir, output_file_name), "w") as file:
            yaml.dump(full_report, file)

        print(f"Metrics saved at {os.path.join(model_dir, output_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gz_evo",
        help="Dataset to be used for evaluation",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained models"
    )
    parser.add_argument(
        "--x_test_path", type=str, required=True, help="Path to the x_test data"
    )
    parser.add_argument(
        "--y_test_path", type=str, required=True, help="Path to the y_test data"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name of the output file for the results",
    )
    parser.add_argument(
        "--model_name", type=str, default="cnn", help="Name of the model to be evaluated"
    )

    args = parser.parse_args()

    main(
        model_dir=args.model_path,
        output_name=args.output_name,
        x_test_path=args.x_test_path,
        y_test_path=args.y_test_path,
        model_name=args.model_name,
        classes=classes_dict[args.dataset],
        dataset=args.dataset,
    )
