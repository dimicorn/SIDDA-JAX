import argparse
import os

import jax.nn as jnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix

from augment import get_transform
from checkpointing import load_models
from dataset import NumpyLoader, classes_dict, dataset_dict


def expected_calibration_error(
    y_true: np.ndarray, y_probs: np.ndarray, num_bins: int = 10
) -> float:
    """Compute the Expected Calibration Error (ECE) for multi-class classification."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_size = 0
        bin_error = 0.0

        for i in range(total_samples):
            prob_pred = y_probs[i, np.argmax(y_probs[i])]

            if bin_lower < prob_pred <= bin_upper:
                bin_size += 1
                is_correct = y_true[i] == np.argmax(y_probs[i])
                bin_error += np.abs(prob_pred - is_correct)

        if bin_size > 0:
            ece += bin_error / total_samples

    return ece


def compute_metrics_with_calibration(
    test_loader: NumpyLoader,
    model,
    model_name: str,
    save_dir: str,
    output_name: str,
    classes: tuple,
) -> tuple:
    """Compute metrics for a model with calibration."""
    y_pred, y_true, feature_maps, y_proba = [], [], [], []

    for imgs, labels in test_loader:
        features, logits = model(imgs, train=False)
        probs = jnn.softmax(logits, axis=1)
        feature_maps.extend(np.asarray(features))
        y_pred.extend(np.asarray(probs.argmax(axis=1)))
        y_proba.extend(np.asarray(probs))
        y_true.extend(np.asarray(labels))

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)

    features_dir = os.path.join(save_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    np.save(
        f"{features_dir}/features_{model_name}_{output_name}.npy", flattened_features
    )

    print("Calibrating classification scores...")
    calibrator = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000), method="sigmoid"
    )
    calibrator.fit(flattened_features, y_true)
    calibrated_proba = calibrator.predict_proba(flattened_features)

    proba_dir = os.path.join(save_dir, "calibrated_probs")
    os.makedirs(proba_dir, exist_ok=True)
    np.save(
        f"{proba_dir}/calibrated_probs_{model_name}_{output_name}.npy", calibrated_proba
    )

    y_pred_calibrated = np.argmax(calibrated_proba, axis=1)
    sklearn_report = classification_report(
        y_true, y_pred_calibrated, output_dict=True, target_names=classes, labels=range(len(classes))
    )

    cf_matrix = confusion_matrix(y_true, y_pred_calibrated, labels=range(len(classes)))
    df_cm = pd.DataFrame(
        cf_matrix / np.maximum(np.sum(cf_matrix, axis=1)[:, None], 1),
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f"{model_name} Calibrated Confusion Matrix")
    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrix")
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            confusion_matrix_dir, f"confusion_matrix_calibrated_{model_name}_{output_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()

    brier_scores = [
        brier_score_loss(y_true == i, calibrated_proba[:, i])
        for i in range(calibrated_proba.shape[1])
    ]
    mean_brier_score = float(np.mean(brier_scores))

    ece = float(expected_calibration_error(y_true, calibrated_proba))

    return sklearn_report, ece, mean_brier_score


def main(
    model_dir: str,
    output_name: str,
    x_test_path: str,
    y_test_path: str,
    model_name: str,
    classes: tuple,
    dataset: str,
) -> None:
    """Main function to evaluate models with calibration."""
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
        model_metrics, ece, brier_score = compute_metrics_with_calibration(
            test_loader=test_dataloader,
            model=model,
            model_name=checkpoint_name,
            save_dir=model_dir,
            output_name=output_name,
            classes=classes,
        )

        model_metrics["ECE"] = ece
        model_metrics["Brier Score"] = brier_score

        print("Compiling Metrics")
        output_file_name = f"{output_name}_{checkpoint_name}.yaml"
        with open(os.path.join(metrics_dir, output_file_name), "w") as file:
            yaml.dump(model_metrics, file)

        print(f"Metrics saved at {os.path.join(metrics_dir, output_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models with calibration")
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
