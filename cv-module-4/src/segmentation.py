"""
CSc 8830 Module 4 - Classical human-boundary segmentation helpers.

Run examples:
    python src/rgb_human_boundary.py --image data/rgb/person.jpg
    python src/thermal_human_boundary.py --image data/thermal/person_thermal.jpg

These functions intentionally use classical OpenCV image processing only:
thresholding, color-space transforms, morphology, edges, contours, GrabCut,
and connected components. No machine learning or deep learning model is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    boundary: np.ndarray
    overlay: np.ndarray
    cutout: np.ndarray
    metrics: dict


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def fill_external_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def keep_components_near_body(mask: np.ndarray) -> np.ndarray:
    """Keep foreground islands that plausibly belong to one standing person."""
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = 1 + int(np.argmax(areas))
    main_x = stats[main_label, cv2.CC_STAT_LEFT]
    main_y = stats[main_label, cv2.CC_STAT_TOP]
    main_w = stats[main_label, cv2.CC_STAT_WIDTH]
    main_h = stats[main_label, cv2.CC_STAT_HEIGHT]
    main_cx = main_x + main_w / 2.0

    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max(20, 0.002 * mask.size):
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        cx = x + bw / 2.0
        horizontally_near = abs(cx - main_cx) < max(main_w * 0.9, mask.shape[1] * 0.12)
        vertically_relevant = y < main_y + main_h + mask.shape[0] * 0.08
        if label == main_label or (horizontally_near and vertically_relevant):
            kept[labels == label] = 255
    return kept


def boundary_from_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, thickness=thickness)
    return boundary


def make_overlay(image: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:, :, 1] = 180
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(image, 0.68, tint, 0.32, 0), overlay)
    overlay[boundary > 0] = (0, 0, 255)
    return overlay


def make_cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def classical_rgb_human_mask(image: np.ndarray) -> np.ndarray:
    """Segment the most likely person in a regular RGB image using OpenCV only."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    blur = cv2.GaussianBlur(lab, (5, 5), 0)
    border = np.concatenate(
        [
            blur[: max(4, h // 25), :, :].reshape(-1, 3),
            blur[-max(4, h // 25) :, :, :].reshape(-1, 3),
            blur[:, : max(4, w // 25), :].reshape(-1, 3),
            blur[:, -max(4, w // 25) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(border, axis=0)
    color_distance = np.linalg.norm(blur.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    _, bg_difference = cv2.threshold(
        color_distance.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bg_difference = cv2.morphologyEx(bg_difference, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bg_difference = cv2.morphologyEx(bg_difference, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    # Color quantization separates foreground-like regions from background texture.
    samples = blur.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.8)
    _, labels, centers = cv2.kmeans(samples, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape((h, w))

    center_x = w / 2.0
    target_area = 0.22 * h * w
    candidates = []
    for label in range(3):
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        component = largest_component(component)
        area = int(np.count_nonzero(component))
        if area < 0.015 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(component)
        border_pixels = (
            np.count_nonzero(component[0, :])
            + np.count_nonzero(component[-1, :])
            + np.count_nonzero(component[:, 0])
            + np.count_nonzero(component[:, -1])
        )
        border_ratio = border_pixels / max(1, 2 * h + 2 * w)
        if border_ratio > 0.18 and area > 0.35 * h * w:
            continue
        moments = cv2.moments(component)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        person_shape_bonus = 0.0
        if bh > 0.35 * h and 0.08 * w < bw < 0.65 * w:
            person_shape_bonus = 0.18 * h * w
        area_penalty = abs(area - target_area)
        center_penalty = abs(cx - center_x) * h
        border_penalty = border_ratio * h * w
        score = person_shape_bonus - 0.55 * area_penalty - 0.28 * center_penalty - 1.4 * border_penalty
        candidates.append((score, component))

    difference_seed = keep_components_near_body(bg_difference)
    if candidates and np.count_nonzero(difference_seed) > 0.02 * h * w:
        seed = cv2.bitwise_or(max(candidates, key=lambda item: item[0])[1], difference_seed)
    elif candidates:
        seed = max(candidates, key=lambda item: item[0])[1]
    else:
        margin_x = max(10, int(0.16 * w))
        margin_y = max(10, int(0.04 * h))
        seed = np.zeros((h, w), np.uint8)
        seed[margin_y : h - margin_y, margin_x : w - margin_x] = 255
    seed = cv2.dilate(seed, np.ones((25, 25), np.uint8), iterations=1)
    sure_fg = cv2.erode(seed, np.ones((13, 13), np.uint8), iterations=1)
    sure_bg = cv2.bitwise_not(cv2.dilate(seed, np.ones((35, 35), np.uint8), iterations=1))

    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[sure_bg > 0] = cv2.GC_BGD
    grabcut_mask[sure_fg > 0] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = keep_components_near_body(mask)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return fill_external_contours(mask)


def classical_thermal_human_mask(image: np.ndarray) -> np.ndarray:
    """Segment a warm human shape in a thermal image using thresholding and contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -3
    )
    mask = cv2.bitwise_or(otsu, adaptive)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = largest_component(mask)
    return fill_external_contours(mask)


def evaluate_against_reference(pred_mask: np.ndarray, ref_mask_path: Optional[str | Path]) -> dict:
    if not ref_mask_path:
        return {}
    reference = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Could not read reference mask: {ref_mask_path}")
    reference = cv2.resize(reference, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = pred_mask > 0
    ref = reference > 0
    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    pred_sum = pred.sum()
    ref_sum = ref.sum()
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float((2 * intersection) / (pred_sum + ref_sum)) if pred_sum + ref_sum else 1.0,
        "predicted_pixels": int(pred_sum),
        "reference_pixels": int(ref_sum),
    }


def segment_and_save(
    image_path: str | Path,
    mode: str,
    output_dir: str | Path = "outputs",
    sam2_mask_path: Optional[str | Path] = None,
) -> SegmentationResult:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)

    if mode == "rgb":
        mask = classical_rgb_human_mask(image)
    elif mode == "thermal":
        mask = classical_thermal_human_mask(image)
    else:
        raise ValueError("mode must be 'rgb' or 'thermal'")

    boundary = boundary_from_mask(mask)
    overlay = make_overlay(image, mask, boundary)
    cutout = make_cutout(image, mask)
    metrics = evaluate_against_reference(mask, sam2_mask_path)

    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_{mode}_mask.png"), mask)
    cv2.imwrite(str(output_dir / f"{stem}_{mode}_boundary.png"), boundary)
    cv2.imwrite(str(output_dir / f"{stem}_{mode}_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / f"{stem}_{mode}_cutout.png"), cutout)
    with (output_dir / f"{stem}_{mode}_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return SegmentationResult(mask=mask, boundary=boundary, overlay=overlay, cutout=cutout, metrics=metrics)


def cli(mode: str) -> None:
    parser = argparse.ArgumentParser(description=f"Classical OpenCV {mode} human boundary extraction")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="outputs", help="Directory for mask, boundary, overlay, and metrics")
    parser.add_argument("--sam2-mask", default=None, help="Optional binary mask exported from SAM2 for comparison")
    args = parser.parse_args()
    result = segment_and_save(args.image, mode, args.output_dir, args.sam2_mask)
    print(json.dumps(result.metrics or {"status": "completed", "sam2_comparison": "not provided"}, indent=2))
