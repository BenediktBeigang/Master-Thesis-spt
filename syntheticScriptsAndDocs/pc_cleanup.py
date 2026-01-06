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
                "filename": output_file,
            },
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    count = pipe.execute()

    end = time.time()
    print(f"{end - start:.2f} sec - {output_file}")


if __name__ == "__main__":
    # folder where all .las files are located
    folder_path = "./data/synthetic/raw"

    # get all .las files in the folder
    las_files = glob.glob(os.path.join(folder_path, "*.las"))

    # target folder to save processed files CHANGE IF THE FILES SHOULD NOT BE OVERWRITTEN
    target_folder = folder_path

    for las_file in las_files:
        las_file_without_extension = os.path.splitext(os.path.basename(las_file))[0]
        output_file = os.path.join(target_folder, f"{las_file_without_extension}.las")
        process(las_file, output_file)
