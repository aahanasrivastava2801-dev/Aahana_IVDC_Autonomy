
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PART B: LANE MARKING AND BACKGROUND ISOLATION
# Method: preprocessing + thresholding + edges + ROI + GrabCut
# No pretrained models are used.
# ============================================================

# 1. Load the road image
image = cv2.imread("road.png")

if image is None:
    raise FileNotFoundError(
        "road.png was not found. Keep it in the same folder as code.py."
    )

height, width = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Threshold bright/white pixels to reveal the lane marking
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

white_mask = cv2.inRange(
    hsv,
    np.array([0, 0, 175], dtype=np.uint8),
    np.array([180, 100, 255], dtype=np.uint8)
)

# 4. Edge detection
edges = cv2.Canny(blurred, 60, 160)

# 5. Define a road-focused region of interest.
# This trapezoid limits processing to the lower road area.
roi_polygon = np.array([[
    (int(0.03 * width), height - 1),
    (width - 1, height - 1),
    (int(0.79 * width), int(0.43 * height)),
    (int(0.48 * width), int(0.43 * height))
]], dtype=np.int32)

roi_mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(roi_mask, roi_polygon, 255)

roi_edges = cv2.bitwise_and(edges, roi_mask)
roi_image = cv2.bitwise_and(image, image, mask=roi_mask)

# 6. Hough line detection for intermediate visualization
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    minLineLength=max(20, int(width * 0.04)),
    maxLineGap=max(10, int(width * 0.03))
)

hough_visualization = image_rgb.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
        cv2.line(
            hough_visualization,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )

# 7. Segment the road region with GrabCut.
# GrabCut uses color differences and foreground/background seeds.
# The ROI polygon provides an initial estimate, not a guaranteed
# exact road boundary.

grabcut_mask = np.full(
    (height, width),
    cv2.GC_BGD,
    dtype=np.uint8
)

# Everything inside the ROI starts as probable foreground.
grabcut_mask[roi_mask > 0] = cv2.GC_PR_FGD

# Use the central lower portion of the road as foreground seeds.
# These seed points must lie on the visible road surface.
seed_polygon = np.array([[
    (int(0.48 * width), int(0.96 * height)),
    (int(0.67 * width), int(0.96 * height)),
    (int(0.59 * width), int(0.67 * height)),
    (int(0.56 * width), int(0.57 * height))
]], dtype=np.int32)

cv2.fillPoly(grabcut_mask, seed_polygon, cv2.GC_FGD)

# Run GrabCut, which estimates foreground and background colors.
background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

try:
    cv2.grabCut(
        image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )
except cv2.error as error:
    raise RuntimeError(
        "GrabCut failed. Check that road.png is a valid color image."
    ) from error

# Keep both definite and probable foreground pixels.
road_candidate = np.where(
    (grabcut_mask == cv2.GC_FGD) |
    (grabcut_mask == cv2.GC_PR_FGD),
    255,
    0
).astype(np.uint8)

# Limit the result to the road-focused ROI.
road_candidate = cv2.bitwise_and(road_candidate, roi_mask)

# Remove small speckles and smooth small gaps.
kernel = np.ones((5, 5), np.uint8)
road_candidate = cv2.morphologyEx(
    road_candidate, cv2.MORPH_CLOSE, kernel, iterations=2
)

# Keep the largest connected foreground component.
count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    road_candidate, connectivity=8
)

road_mask = np.zeros((height, width), dtype=np.uint8)

if count > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    road_mask[labels == largest_label] = 255

# 8. Preserve the white lane marking inside the segmented road.
# The final output follows the judge's example:
# road surface = black, surroundings = white, marking = white.
lane_marking_inside_road = cv2.bitwise_and(white_mask, road_mask)

# Black road pixels, white background.
final_bitmap = cv2.bitwise_not(road_mask)

# Keep detected bright marking white within the black road region.
final_bitmap[lane_marking_inside_road > 0] = 255

# 9. Overlay the segmented road on the original image for inspection.
overlay = image_rgb.copy()

# Mark the estimated road region with a translucent blue tint.
road_pixels = road_mask > 0
overlay[road_pixels] = (
    0.65 * overlay[road_pixels] +
    0.35 * np.array([50, 120, 255])
).astype(np.uint8)

# 10. Display all pipeline stages
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

panels = [
    (image_rgb, "1. Original Image", None),
    (gray, "2. Grayscale", "gray"),
    (blurred, "3. Gaussian Blur", "gray"),
    (edges, "4. Canny Edge Detection", "gray"),
    (white_mask, "5. White Marking Threshold", "gray"),
    (roi_edges, "6. Edges Inside ROI", "gray"),
    (hough_visualization, "7. Hough Line Segments", None),
    (overlay, "8. Estimated Road Region", None),
    (final_bitmap, "9. Final Bitmap", "gray"),
]

for ax, (picture, title, cmap) in zip(axes, panels):
    if cmap == "gray":
        ax.imshow(picture, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(picture)

    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout(pad=2)
plt.savefig("lane_detection_results.png", dpi=200, bbox_inches="tight")
plt.show()
# 11. Save output files
cv2.imwrite("road_mask.png", road_mask)
cv2.imwrite("final_bitmap.png", final_bitmap)
cv2.imwrite("roi_edges.png", roi_edges)
cv2.imwrite("white_marking_threshold.png", white_mask)
cv2.imwrite("grabcut_candidate.png", road_candidate)
cv2.imwrite("estimated_road_overlay.png", cv2.cvtColor(
    overlay, cv2.COLOR_RGB2BGR
))

print("Processing complete.")
print("Saved final output: final_bitmap.png")
print("Saved road mask: road_mask.png")
print("Saved intermediate images and lane_detection_results.png")
print("Inspect Panel 7 and Panel 8 to verify the estimated road boundary.")