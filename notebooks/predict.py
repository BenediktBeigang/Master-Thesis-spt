import sys

sys.path.append("..")
import os
import argparse
import torch
from src.datasets.synthetic_config import (
    CLASS_NAMES,
    CLASS_COLORS,
    ID2TRAINID,
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


def predict(filepath, checkpoint):
    data = read_synthetic(filepath, semantic=False, remap=False)

    cfg = init_config(
        overrides=[
            f"experiment=semantic/synthetic_11g",
            f"datamodule.load_full_res_idx=True",
        ]
    )
    cfg.keys()

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
    checkpoint_identifier = checkpoint.split("/")[-1].split("_")
    output_filename = filepath.replace(
        ".las", f"_predicted_{checkpoint_identifier[0]}_t{checkpoint_identifier[2]}.las"
    )
    output_las.write(output_filename)
    print(f"Saved predictions to: {output_filename}")


if __name__ == "__main__":
    """
    Use it with:
    python predict.py --file path/to/your/file.las --checkpoint path/to/your
    """
    parser = argparse.ArgumentParser(
        description="LAS-Dateien nach PointSourceId filtern"
    )
    parser.add_argument(
        "--file", "-f", help="Einzelne LAS-Datei verarbeiten", required=False
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="Pfad zum Checkpoint für die Klassifizierung",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Fehler: Checkpoint nicht gefunden {args.checkpoint}")
        exit(1)

    if not args.file:
        path = [
            "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_0/ontras_0.las",
            "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_1/ontras_1.las",
            "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_2/ontras_2.las",
            "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_3/ontras_3.las",
            "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_4/ontras_4.las",
            #    "/home/benedikt/Documents/repos/superpoint_transformer/data/ontras/ontras_5/ontras_5.las",
        ]

    if args.file:
        start_time = time.time()
        predict(args.file, args.checkpoint)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Prediction completed in {total_time:.2f} seconds")
    else:
        for file in path:
            start_time = time.time()
            predict(file, args.checkpoint)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Prediction for {file} completed in {total_time:.2f} seconds")
