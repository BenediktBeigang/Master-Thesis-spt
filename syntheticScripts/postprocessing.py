import pdal
import json
import argparse
import os
import time
import glob

radius = 0.02


def process(input_file, output_file):
    start = time.time()

    if not os.path.exists(input_file):
        print(f"Fehler: Datei {input_file} nicht gefunden")
        return

    pipeline = {
        "pipeline": [
            input_file,
            {"type": "filters.sample", "radius": radius},
            {
                "type": "filters.expression",
                "expression": "!((PointSourceId == 5 && Z < -3) || Z > 50)",
            },
            {
                "type": "writers.las",
                "filename": output_file,
            },
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    count = pipe.execute()

    end = time.time()
    print(f"{end - start:.2f} sec - {output_file}")


if __name__ == "__main__":
    # folder_path = "/home/benedikt/Documents/repos/pointcloud-dataset/synthetic/raw"
    folder_path = "/media/benedikt/50140 LOS3 Brandenburg Nord/Synthetic/raw/800"
    # folder_path = (
    #     "/home/benedikt/Documents/repos/pointcloud-dataset/synthetic/cleaned/0700"
    # )
    # target_folder = (
    #     "/home/benedikt/Documents/repos/pointcloud-dataset/synthetic/cleaned/0800"
    # )
    # las_files = glob.glob(os.path.join(folder_path, "*.las"))

    las_files = [
        "/mnt/c/Users/bened/Documents/GitHub/gds/output/810_202510210725_frames_1_to_1059_noise_parts.las",
        "/mnt/c/Users/bened/Documents/GitHub/gds/output/811_202510210744_frames_1_to_1059_noise_parts.las",
    ]
    target_folder = "/mnt/c/Users/bened/Documents/GitHub/gds/output"

    for las_file in las_files:
        las_file_without_extension = os.path.splitext(os.path.basename(las_file))[0]
        output_file = os.path.join(target_folder, f"{las_file_without_extension}.las")
        process(las_file, output_file)
