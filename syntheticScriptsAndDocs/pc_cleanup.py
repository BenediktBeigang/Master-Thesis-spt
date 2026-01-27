import pdal
import json
import os
import time
import glob

RADIUS = 0.02  # space between points in meters


def process(input_file, output_file):
    start = time.time()

    if not os.path.exists(input_file):
        print(f"Fehler: Datei {input_file} nicht gefunden")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pipeline = {
        "pipeline": [
            input_file,
            {"type": "filters.sample", "radius": RADIUS},
            {
                "type": "filters.expression",
                "expression": "!((PointSourceId == 5 && Z < -3) || Z > 50)",
            },
            {
                "type": "writers.las",
                # "compression": "true",
                "filename": output_file,
            },
        ]
    }

    try:
        pipe = pdal.Pipeline(json.dumps(pipeline))
        count = pipe.execute()
        print(f"{time.time() - start:.2f} sec - {count} points - {output_file}")
    except RuntimeError as e:
        print(f"PDAL Fehler bei {input_file}: {e}")


if __name__ == "__main__":
    # folder where all .las files are located
    folder_path = "/mnt/i/Synthetic/raw/800"

    # get all .las files in the folder
    las_files = glob.glob(os.path.join(folder_path, "*.las"))

    # target folder to save processed files CHANGE IF THE FILES SHOULD NOT BE OVERWRITTEN
    target_folder = "/mnt/c/Users/benedikt.beigang/Nextcloud/Team Geospatial/Produkte/Geodatenserver/Punktwolken/Synthetic/0800"

    for las_file in las_files:
        las_file_without_extension = os.path.splitext(os.path.basename(las_file))[0]
        output_file = os.path.join(target_folder, f"{las_file_without_extension}.laz")
        process(las_file, output_file)
