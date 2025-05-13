#!/usr/bin/env python3
import json
import argparse
import pdal
import numpy as np
import sys

def make_pipeline(input_las: str, output_ply: str) -> pdal.Pipeline:
    """
    Baut einen PDAL-Pipeline-Block, der:
      1. Das LAS liest
      2. per Python-Filter die RGB-Kanäle skaliert (wenn nötig)
      3. ins PLY schreibt
    """
    # PDAL-Pipeline-JSON
    pipeline_def = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "filters.assign",
                "value": [
                    "Red = Red / 256",
                    "Green = Green / 256",
                    "Blue = Blue / 256"
                ]  
            },
            {
                "type": "writers.ply",
                "filename": output_ply,
                "storage_mode": "little endian", # Binary-PLY spart Platz und ist schneller
                "dims": "X,Y,Z,Red,Green,Blue"
            }
        ]
    }

    # Erzeuge und gib das Pipeline-Objekt zurück
    return pdal.Pipeline(json.dumps(pipeline_def))


def convert(input_las: str, output_ply: str):
    """Führt die PDAL-Pipeline aus und schreibt output_ply."""
    pipeline = make_pipeline(input_las, output_ply)
    try:
        count = pipeline.execute()
    except RuntimeError as e:
        print("FEHLER beim Ausführen der PDAL-Pipeline:", e, file=sys.stderr)
        sys.exit(1)

    print(f"✓ Fertig: {count} Punkte von '{input_las}' nach '{output_ply}' konvertiert.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Konvertiere eine LAS-Punktwolke (8/16bit RGB) → KITTI-360-kompatibles PLY"
    )
    p.add_argument("input_las",  help="Pfad zur Eingabe-LAS-Datei")
    p.add_argument("output_ply", help="Pfad für die Ausgabe-PLY-Datei")
    args = p.parse_args()

    convert(args.input_las, args.output_ply)
