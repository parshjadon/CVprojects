"""
CSc 8830 Module 4 - RGB human boundary extraction.

How to run:
    python src/rgb_human_boundary.py --image data/rgb/person.jpg
    python src/rgb_human_boundary.py --image data/rgb/person.jpg --sam2-mask outputs/person_sam2_mask.png

Outputs are written to outputs/:
    *_rgb_mask.png, *_rgb_boundary.png, *_rgb_overlay.png, *_rgb_cutout.png, *_rgb_metrics.json
"""

from segmentation import cli


if __name__ == "__main__":
    cli("rgb")
