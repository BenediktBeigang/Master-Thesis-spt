import os
import argparse
import torch
from src.datasets.synthetic_config import (
    CLASS_NAMES,
    SYNTHETIC_NUM_CLASSES,
)
from src.datasets.synthetic import read_synthetic
from src.utils import init_config
from src.transforms import instantiate_datamodule_transforms
from src.transforms import NAGRemoveKeys
import hydra
import numpy as np
import laspy
import time


def predict(filepath: str, checkpoint: str, output_path=None):
    start = time.time()
    data = read_synthetic(filepath, semantic=False, remap=False)

    cfg = init_config(
        overrides=[
            f"experiment=semantic/synthetic_11g",
            f"datamodule.load_full_res_idx=True",
        ]
    )
    cfg.keys()

    time_config = time.time()

    print(f"  SYNTHETIC_NUM_CLASSES: {SYNTHETIC_NUM_CLASSES}")
    print(f"Config num_classes: {cfg.datamodule.num_classes}")

    transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)
    nag = transforms_dict["pre_transform"](data)

    nag = NAGRemoveKeys(
        level=0,
        keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys],
    )(nag)
    nag = NAGRemoveKeys(
        level="1+",
        keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys],
    )(nag)
    nag = nag.cuda()
    nag = transforms_dict["on_device_test_transform"](nag)

    time_preTransform = time.time()

    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(checkpoint)

    model = model.eval().to(nag.device)
    print(nag)
    with torch.no_grad():
        output = model(nag)

    print(output.semantic_pred().shape)
    print(nag.num_points)

    # Compute full-resolution semantic predictions
    raw_semseg_y = output.full_res_semantic_pred(
        super_index_level0_to_level1=nag[0].super_index, sub_level0_to_raw=nag[0].sub
    )

    time_inference = time.time()

    print(f"Full resolution predictions shape: {raw_semseg_y.shape}")
    print(f"Original data points: {data.num_points}")

    original_las = laspy.read(filepath)
    assert len(raw_semseg_y) == len(
        original_las.points
    ), f"Mismatch: {len(raw_semseg_y)} predictions vs {len(original_las.points)} points"

    # Neue LAS-Datei erstellen mit Predictions
    # Kopiere die ursprüngliche Struktur
    output_las = laspy.LasData(original_las.header)
    output_las.points = original_las.points

    # Füge die semantischen Predictions als neues Feld hinzu
    # Konvertiere zu numpy array falls es ein torch tensor ist
    if hasattr(raw_semseg_y, "cpu"):
        predictions = raw_semseg_y.cpu().numpy().astype(np.uint8)
    else:
        predictions = np.array(raw_semseg_y, dtype=np.uint8)

    # Klassifizierungsfeld setzen
    output_las.classification = predictions

    # Optional: Auch die Klassennamen für bessere Interpretierbarkeit hinzufügen
    print("Predicted classes:")
    unique_classes = np.unique(predictions)
    for cls in unique_classes:
        count = np.sum(predictions == cls)
        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Unknown_{cls}"
        print(f"  Class {cls} ({class_name}): {count} points")

    # Ausgabedatei speichern
    output_path = output_path if output_path is not None else filepath
    output_las.write(output_path)

    time_save = time.time()
    return start, time_config, time_preTransform, time_inference, time_save


if __name__ == "__main__":
    """
    Use it with:
    python predict.py --file path/to/your/file.las --checkpoint path/to/your/checkpoint.ckpt
    """
    parser = argparse.ArgumentParser(
        description="LAS-Dateien nach PointSourceId filtern"
    )
    parser.add_argument(
        "--file", "-f", help="Einzelne LAS-Datei verarbeiten", required=True
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="Pfad zum Checkpoint für die Klassifizierung",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        "-o",
        help="Pfad zum Speichern der vorhergesagten LAS-Datei (optional, Standard: Datei wird überschrieben)",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Fehler: LAS-Datei nicht gefunden {args.file}")
        exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"Fehler: Checkpoint nicht gefunden {args.checkpoint}")
        exit(1)

    start, time_config, time_preTransform, time_inference, time_save = predict(
        args.file, args.checkpoint, args.output_path
    )

    print(f"File: {args.file}")
    print(f"   Config time: {time_config - start:.2f} seconds")
    print(f"   Pre-transform time: {time_preTransform - time_config:.2f} seconds")
    print(f"   Inference time: {time_inference - time_preTransform:.2f} seconds")
    print(f"   Save time: {time_save - time_inference:.2f} seconds")
    print(f"=> Total time: {time_save - start:.2f} seconds")
    print("--------------------------------------------------")
