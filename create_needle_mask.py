import os
import argparse
import cv2
import numpy as np


def parse_roi(roi_str):
    if roi_str is None:
        return None
    vals = [int(v) for v in roi_str.replace(",", " ").split()]
    if len(vals) != 4:
        raise ValueError("ROI must be four integers: x1 y1 x2 y2")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="/home/xsun97/FoundationPose/AMBF_data/rgb/cameraR_000000_1777481066520966144.png",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/xsun97/FoundationPose/AMBF_data/mask",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optional ROI: x1 y1 x2 y2. Example: --roi '430 120 640 420'",
    )
    parser.add_argument("--s_max", type=int, default=80)
    parser.add_argument("--v_min", type=int, default=145)
    parser.add_argument("--min_area", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    H, W = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # White needle mask:
    # white / light gray usually has low saturation and high value.
    lower_white = np.array([0, 0, args.v_min], dtype=np.uint8)
    upper_white = np.array([179, args.s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Optional ROI to avoid picking up unrelated bright regions.
    roi = parse_roi(args.roi)
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(W, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1))
        y2 = max(0, min(H, y2))

        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    # Clean mask.
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove tiny components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= args.min_area:
            clean[labels == label] = 255

    mask = clean

    # Save binary mask.
    mask_path = os.path.join(args.out_dir, "needle_mask.png")
    cv2.imwrite(mask_path, mask)

    # Save masked needle crop/result.
    needle_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    needle_only_path = os.path.join(args.out_dir, "needle_only.png")
    cv2.imwrite(needle_only_path, needle_only)

    # Save overlay for checking.
    overlay = img_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

    overlay_path = os.path.join(args.out_dir, "needle_mask_overlay.png")
    cv2.imwrite(overlay_path, blended)

    print(f"Saved binary mask: {mask_path}")
    print(f"Saved needle-only image: {needle_only_path}")
    print(f"Saved overlay image: {overlay_path}")
    print(f"Mask pixels: {np.count_nonzero(mask)}")


if __name__ == "__main__":
    main()
