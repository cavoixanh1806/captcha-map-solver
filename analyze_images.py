"""Quantitative analysis of CAPTCHA images to inform model design."""
import csv, os, json, hashlib
from collections import Counter
import numpy as np
import cv2

DATA = r"c:\Users\Administrator\Desktop\TrainAI\data"
META = os.path.join(DATA, "metadata.csv")

rows = list(csv.DictReader(open(META)))

# 1) Detect duplicate IMAGE files (by md5 of pixels), independent of label
hashes = {}
dup_pairs = []
for r in rows:
    p = os.path.join(DATA, r["filename"])
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in hashes:
        dup_pairs.append((hashes[h], r["filename"], r["text"]))
    else:
        hashes[h] = r["filename"]

# 2) Text bounding box estimation
# Strategy: convert to LAB, take L channel, the text pixels are typically saturated colors
# unlike pastel background. We can use saturation (HSV S) > threshold to mask glyph pixels.
boxes = []
for r in rows:
    p = os.path.join(DATA, r["filename"])
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mask = sat > 80  # saturated foreground glyph pixels
    if mask.sum() < 50:
        continue
    ys, xs = np.where(mask)
    boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), int(mask.sum())))

boxes_arr = np.array(boxes)
xmin = boxes_arr[:, 0]
ymin = boxes_arr[:, 1]
xmax = boxes_arr[:, 2]
ymax = boxes_arr[:, 3]
widths = xmax - xmin
heights = ymax - ymin

# 3) Color analysis of glyph pixels
all_colors = []
for r in rows[:100]:
    p = os.path.join(DATA, r["filename"])
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > 80
    pixels = img[mask]
    all_colors.append(pixels)
fg = np.concatenate(all_colors)
print("=== TEXT REGION STATS ===")
print(f"images analyzed: {len(boxes)}")
print(f"x_min  : min={xmin.min()} mean={xmin.mean():.1f} max={xmin.max()}")
print(f"y_min  : min={ymin.min()} mean={ymin.mean():.1f} max={ymin.max()}")
print(f"x_max  : min={xmax.min()} mean={xmax.mean():.1f} max={xmax.max()}")
print(f"y_max  : min={ymax.min()} mean={ymax.mean():.1f} max={ymax.max()}")
print(f"width  : min={widths.min()} mean={widths.mean():.1f} max={widths.max()}")
print(f"height : min={heights.min()} mean={heights.mean():.1f} max={heights.max()}")

print("\n=== PIXEL DUPLICATES ===")
print(f"unique images: {len(hashes)} / {len(rows)} → duplicates: {len(dup_pairs)}")
for a, b, txt in dup_pairs[:10]:
    print(f"  {a} == {b}  (label='{txt}')")

print("\n=== FOREGROUND COLOR (HSV) ===")
fg_hsv = cv2.cvtColor(fg.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
print(f"H: mean={fg_hsv[:,0].mean():.1f} std={fg_hsv[:,0].std():.1f}")
print(f"S: mean={fg_hsv[:,1].mean():.1f} std={fg_hsv[:,1].std():.1f}")
print(f"V: mean={fg_hsv[:,2].mean():.1f} std={fg_hsv[:,2].std():.1f}")

# 4) Edge density (proxy for noise level)
edge_ratios = []
for r in rows[:200]:
    p = os.path.join(DATA, r["filename"])
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    edge_ratios.append(edges.mean() / 255)
print(f"\n=== EDGE DENSITY (Canny) ===")
print(f"mean edge ratio: {np.mean(edge_ratios):.4f}  std: {np.std(edge_ratios):.4f}")

# 5) Save tight-box stats per image to inform crop strategy
percentiles = lambda a: f"p1={np.percentile(a,1):.0f} p5={np.percentile(a,5):.0f} p50={np.percentile(a,50):.0f} p95={np.percentile(a,95):.0f} p99={np.percentile(a,99):.0f}"
print(f"\nx_min   {percentiles(xmin)}")
print(f"y_min   {percentiles(ymin)}")
print(f"x_max   {percentiles(xmax)}")
print(f"y_max   {percentiles(ymax)}")
print(f"width   {percentiles(widths)}")
print(f"height  {percentiles(heights)}")
