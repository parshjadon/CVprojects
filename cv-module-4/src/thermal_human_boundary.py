"""
CSc 8830 Module 4 - Thermal human boundary extraction.

How to run:
    python src/thermal_human_boundary.py --image data/thermal/person_thermal.jpg
    python src/thermal_human_boundary.py --image data/thermal/person_thermal.jpg --sam2-mask outputs/person_thermal_sam2_mask.png

Outputs are written to outputs/:
    *_thermal_mask.png, *_thermal_boundary.png, *_thermal_overlay.png, *_thermal_cutout.png, *_thermal_metrics.json
"""

from segmentation import cli


if __name__ == "__main__":
    cli("thermal")
