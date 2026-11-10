#!/usr/bin/env python3
"""
Main Evaluation Script for Plant Species and Health Classification
Evaluates:
  1. Species Classifier: Acacia senegal vs. Prosopis cineraria (EfficientNetB0)
  2. PC Health Classifier: Healthy vs. Unhealthy (FT-EfficientNetB0 + Tabular Data + RBF-SVM)
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate plant classification models on unseen test sets.")
parser.add_argument("--cpu", action="store_true", help="Force CPU-only execution (disables GPU)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for CNN inference (default: 16)")
parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory (default: ./data)")
args = parser.parse_args()

# ── Environment & Hardware Setup ──────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = args.data_dir if args.data_dir else os.path.join(SCRIPT_DIR, "data")

print("=" * 88)
print("PLANT SPECIES & HEALTH CLASSIFICATION: TEST EVALUATION")
print("=" * 88)

# Detect runtime hardware
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and not args.cpu:
        print(f"[Device] Using GPU: {gpus[0].name}")
    else:
        print("[Device] Using CPU mode")
except Exception:
    print("[Device] Using CPU mode")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "specificity": spec,
        "cm": cm
    }

def print_metrics(model_title, test_info, metrics):
    print("\n" + "=" * 88)
    print(model_title)
    print(f"Test Split: {test_info}")
    print("-" * 88)
    print(f"  Accuracy:         {metrics['accuracy'] * 100:.2f}%  ({metrics['accuracy']:.6f})")
    print(f"  ROC-AUC:          {metrics['auc']:.4f}    ({metrics['auc']:.6f})")
    print(f"  Precision:        {metrics['precision'] * 100:.2f}%  ({metrics['precision']:.6f})")
    print(f"  Recall (Sens.):   {metrics['recall'] * 100:.2f}%  ({metrics['recall']:.6f})")
    print(f"  Specificity:      {metrics['specificity'] * 100:.2f}%  ({metrics['specificity']:.6f})")
    print(f"  F1-Score:         {metrics['f1'] * 100:.2f}%  ({metrics['f1']:.6f})")
    print("  Confusion Matrix:")
    print(f"    [[{metrics['cm'][0][0]:>5}, {metrics['cm'][0][1]:>5}],")
    print(f"     [{metrics['cm'][1][0]:>5}, {metrics['cm'][1][1]:>5}]]")
    print("=" * 88)


# ═════════════════════════════════════════════════════════════════════════════
# 1. MODEL 1: SPECIES CLASSIFIER (Acacia senegal vs. Prosopis cineraria)
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_species_classifier():
    print("\nEvaluating Model 1: Species Classifier (EfficientNetB0)...")
    p1 = os.path.join(DATA_DIR, "as_pc_test_part1.npz")
    p2 = os.path.join(DATA_DIR, "as_pc_test_part2.npz")
    weights_path = os.path.join(SCRIPT_DIR, "as_pc_classifier", "best_weights.weights.h5")

    if not (os.path.exists(p1) and os.path.exists(p2)):
        print(f"[Error] Missing test data in {DATA_DIR}. Please ensure as_pc_test_part1.npz and as_pc_test_part2.npz exist.")
        return

    d1 = np.load(p1)
    d2 = np.load(p2)
    X_test = np.concatenate([d1["X"], d2["X"]])
    y_test = np.concatenate([d1["y"], d2["y"]])

    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    inp = layers.Input(shape=(224, 224, 3), name="image_input")
    x = layers.Lambda(lambda img: preprocess_input(img), name="preprocess")(inp)
    bb = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    bb.trainable = False
    x = bb(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc_head")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=inp, outputs=out, name="efficientnetb0_species_classifier")

    model.load_weights(weights_path)

    probs = model.predict(X_test, batch_size=args.batch_size, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    n_as = int(np.sum(y_test == 0))
    n_pc = int(np.sum(y_test == 1))
    info = f"{len(y_test)} samples ({n_as} Acacia senegal, {n_pc} Prosopis cineraria)"
    metrics = compute_metrics(y_test, preds, probs)
    print_metrics("MODEL 1: SPECIES CLASSIFIER (Acacia senegal vs. Prosopis cineraria)", info, metrics)


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL 2: PC HEALTH CLASSIFIER (FT-EfficientNetB0 + Tabular Data + RBF-SVM)
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_pc_health_classifier():
    print("\nEvaluating Model 2: PC Health Classifier (FT-EfficientNetB0 + Tabular Data + RBF-SVM)...")
    test_path = os.path.join(DATA_DIR, "pc_health_test.npz")
    svm_path = os.path.join(SCRIPT_DIR, "pc_health_classifier", "svm_model.pkl")
    scaler_path = os.path.join(SCRIPT_DIR, "pc_health_classifier", "combined_scaler.pkl")

    if not os.path.exists(test_path):
        print(f"[Error] Missing test data in {test_path}.")
        return

    if not os.path.exists(svm_path):
        print(f"[Warning] Required model artifact missing: {svm_path}")
        print("Download the svm_model.pkl asset from the project's GitHub Release and place it at this path.")
        print("Skipping PC health classifier evaluation.")
        return

    sign = -1

    data = np.load(test_path)
    cnn_features = data["cnn_features"]
    tab_data = data["tab_data"]
    y_test = data["y_test"]

    X_comb = np.concatenate([cnn_features, tab_data], axis=1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_comb)

    with open(svm_path, "rb") as f:
        clf = pickle.load(f)

    raw_scores = np.asarray(clf.decision_function(X_scaled), dtype=np.float64).ravel() * sign
    preds = (raw_scores >= 0).astype(int)

    n_h = int(np.sum(y_test == 0))
    n_u = int(np.sum(y_test == 1))
    info = f"{len(y_test)} samples ({n_h} Healthy, {n_u} Unhealthy)"
    metrics = compute_metrics(y_test, preds, raw_scores)
    print_metrics("MODEL 2: PC HEALTH CLASSIFIER (Fine-Tuned EfficientNetB0 + Tabular Data + RBF-SVM)", info, metrics)


if __name__ == "__main__":
    evaluate_species_classifier()
    evaluate_pc_health_classifier()
    print("\nEvaluation run finished; review any warnings above.\n")
