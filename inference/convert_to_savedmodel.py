import argparse
import json
import os
import tarfile
import tempfile

import joblib
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, help="Example: artifacts/versioned/v1.0.0")
    args = ap.parse_args()

    artifact_dir = args.artifact_dir.rstrip("/\\")
    meta_path = os.path.join(artifact_dir, "metadata.json")
    keras_dir = os.path.join(artifact_dir, "keras_model")
    preprocess_path = os.path.join(artifact_dir, "preprocess.joblib")
    label_map_path = os.path.join(artifact_dir, "label_map.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    if not os.path.isdir(keras_dir):
        raise FileNotFoundError(f"Missing {keras_dir}")
    if not os.path.exists(preprocess_path):
        raise FileNotFoundError(f"Missing {preprocess_path}")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Missing {label_map_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load model + preprocess (preprocess and label map are packed alongside SavedModel)
    model = tf.keras.models.load_model(keras_dir)
    _ = joblib.load(preprocess_path)  # validate it loads
    with open(label_map_path, "r", encoding="utf-8") as f:
        _ = json.load(f)

    out_tar = os.path.join(artifact_dir, "saved_model.tar.gz")

    # Export SavedModel and package it
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = os.path.join(tmp, "saved_model")
        tf.saved_model.save(model, export_dir)

        with tarfile.open(out_tar, "w:gz") as tar:
            # SavedModel (required by TF Serving)
            tar.add(export_dir, arcname="saved_model")

            # Keep preprocessing + label map inside tar for later use (optional, but practical)
            tar.add(preprocess_path, arcname="preprocess.joblib")
            tar.add(label_map_path, arcname="label_map.json")

            # Include metadata too (useful for debugging/traceability)
            tar.add(meta_path, arcname="metadata.json")

    print("CONVERT_OK")
    print("MODEL_VERSION:", meta.get("model_version"))
    print("WROTE:", out_tar)


if __name__ == "__main__":
    main()
