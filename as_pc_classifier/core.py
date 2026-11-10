"""Shared model training core for AS-vs-PC classification."""

import json
import logging
import os
import sys

import numpy as np

from config import BATCH_SIZE, EPOCHS, LR, PATIENCE, SEED, SPLIT_FILE


def _setup_runtime_env():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if conda_lib not in current_ld:
            os.environ["LD_LIBRARY_PATH"] = conda_lib + (":" + current_ld if current_ld else "")


def _make_logger(log_path, name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _get_splits(y):
    from sklearn.model_selection import train_test_split

    if os.path.exists(SPLIT_FILE):
        data = np.load(SPLIT_FILE)
        idx_tr, idx_v, idx_te = data["idx_train"], data["idx_val"], data["idx_test"]
        return idx_tr, idx_v, idx_te

    idx = np.arange(len(y))
    idx_tv, idx_te = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y)
    idx_tr, idx_v = train_test_split(
        idx_tv,
        test_size=round(0.15 / 0.85, 8),
        random_state=SEED,
        stratify=y[idx_tv],
    )
    np.savez(SPLIT_FILE, idx_train=idx_tr, idx_val=idx_v, idx_test=idx_te)
    return idx_tr, idx_v, idx_te


def _build_model(backbone):
    import tensorflow as tf
    from tensorflow.keras import Model, layers
    from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
    from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
    from tensorflow.keras.optimizers import Adam

    if backbone == "DenseNet121":
        backbone_ctor = DenseNet121
        preprocess_input = dense_preprocess
    elif backbone == "EfficientNetB0":
        backbone_ctor = EfficientNetB0
        preprocess_input = eff_preprocess
    elif backbone == "ResNet50":
        backbone_ctor = ResNet50
        preprocess_input = res_preprocess
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    inp = layers.Input(shape=(224, 224, 3), name="image_input")
    x = layers.Lambda(lambda img: preprocess_input(img), name="preprocess")(inp)
    bb = backbone_ctor(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    bb.trainable = False
    x = bb(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc_head")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=inp, outputs=out, name=f"{backbone.lower()}_as_pc_classifier")
    model.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def _evaluate(model, X_te, y_te, results_file, logger, backbone, y_train_len, y_val_len):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    probs = model.predict(X_te, batch_size=BATCH_SIZE, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_te, preds).tolist()
    results = {
        "model": backbone,
        "backbone": backbone,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs_cap": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "n_train": int(y_train_len),
        "n_val": int(y_val_len),
        "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_te, probs)),
        "confusion_matrix": cm,
    }
    logger.info("Results: %s", results)
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def train_backbone(backbone, script_dir):
    _setup_runtime_env()

    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight

    from preprocess import load_and_preprocess

    log = _make_logger(os.path.join(script_dir, "train.log"), backbone.lower())
    log.info("TF %s  GPUs: %s", tf.__version__, [g.name for g in tf.config.list_physical_devices("GPU")])

    X, y, _ = load_and_preprocess(logger=log)

    idx_tr, idx_v, idx_te = _get_splits(y)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_v, y_v = X[idx_v], y[idx_v]
    X_te, y_te = X[idx_te], y[idx_te]
    log.info("Loaded splits — train=%d val=%d test=%d", len(idx_tr), len(idx_v), len(idx_te))

    cw_vals = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(cw_vals[0]), 1: float(cw_vals[1])}
    log.info("Class weights: %s", class_weight)

    model = _build_model(backbone)
    ckpt_json = os.path.join(script_dir, "checkpoint.json")
    ckpt_weights = os.path.join(script_dir, "best_weights.weights.h5")
    results_file = os.path.join(script_dir, "results.json")

    initial_epoch = 0
    if os.path.exists(ckpt_json) and os.path.exists(ckpt_weights):
        with open(ckpt_json) as fh:
            ck = json.load(fh)
        if ck.get("training_done"):
            log.info("Training complete. Evaluating only.")
            model.load_weights(ckpt_weights)
            return _evaluate(model, X_te, y_te, results_file, log, backbone, len(idx_tr), len(idx_v))
        initial_epoch = int(ck.get("last_epoch", 0))
        model.load_weights(ckpt_weights)
        log.info("Resuming from epoch %d", initial_epoch)

    class CkptJSON(Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(ckpt_json, "w") as fh:
                json.dump({"last_epoch": epoch + 1, "training_done": False}, fh)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_weights, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        CkptJSON(),
    ]
    log.info("Training epochs %d->%d  batch=%d lr=%s", initial_epoch, EPOCHS, BATCH_SIZE, LR)
    try:
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_v, y_v),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
    except tf.errors.ResourceExhaustedError:
        log.error("OOM! Reduce BATCH_SIZE and restart.")
        sys.exit(1)

    with open(ckpt_json, "w") as fh:
        json.dump({"last_epoch": EPOCHS, "training_done": True}, fh)

    if os.path.exists(ckpt_weights):
        model.load_weights(ckpt_weights)

    return _evaluate(model, X_te, y_te, results_file, log, backbone, len(idx_tr), len(idx_v))
