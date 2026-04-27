import sys
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

# ==========================================
# CLI ARGUMENT PARSING
# ==========================================
if len(sys.argv) < 3:
    print("Usage: python3 fleet_inspector.py <path_to_macro_trailer_image> <path_to_micro_tape_image>")
    sys.exit(1)

MACRO_RAW_PATH = sys.argv[1]
MICRO_RAW_PATH = sys.argv[2]

if not os.path.exists(MACRO_RAW_PATH) or not os.path.exists(MICRO_RAW_PATH):
    print("❌ Error: One or both input images could not be found.")
    sys.exit(1)

# --- MODEL SETTINGS ---
POSE_WEIGHTS_PATH = 'trailer_best.pt' # Model for finding the trailer side
LOGO_WEIGHTS_PATH = 'logo_best.pt'   # Model for detecting brands

try:
    print("Loading AI Models...")
    trailer_model = YOLO(POSE_WEIGHTS_PATH)
    logo_model = YOLO(LOGO_WEIGHTS_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO models: {e}")
    print("Make sure 'trailer_pose_best.pt' and 'logo_brand_best.pt' are in the correct location.")
    sys.exit(1)


# ==========================================
# CORE PIPELINE FUNCTIONS
# ==========================================

def flatten_trailer_side(image_path, yolo_model):
    img = cv2.imread(image_path)
    if img is None: return None

    results = yolo_model(img, verbose=False)[0]

    if len(results.boxes) == 0:
        print("❌ YOLO found no trailer side.")
        return None

    box = results.boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    width, height = x2 - x1, y2 - y1
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flat_img = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imwrite("proof_1_flattened_trailer.jpg", flat_img)
    return flat_img

def classify_tape_pattern(micro_image_path):
    img = cv2.imread(micro_image_path)
    if img is None: return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 80, 40]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 40]), np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    temp_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3] * 1.5]

    if not temp_boxes:
        print("❌ No red strips found for classification.")
        return None

    max_width = max([b[2] for b in temp_boxes])
    boxes = sorted([b for b in temp_boxes if b[2] > max_width * 0.5], key=lambda b: b[0])

    if len(boxes) < 2:
        print("❌ Need at least two clean red strips to measure the gap.")
        return None

    x1, y1, w1, h1 = boxes[0]
    x2, y2, w2, h2 = boxes[1]

    gap_width = x2 - (x1 + w1)
    if gap_width <= 0: return None

    ratio = w1 / gap_width
    pattern = 12 if ratio < 1.3 else 18

    # Visual Proof Generation
    debug_img = img.copy()
    cv2.rectangle(debug_img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 3)
    cv2.rectangle(debug_img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 3)
    cv2.line(debug_img, (x1+w1, int(y1+h1/2)), (x2, int(y1+h1/2)), (255, 0, 0), 4)
    cv2.putText(debug_img, f"Class: {pattern}-Inch (Ratio {ratio:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv2.imwrite("proof_2_micro_classification.jpg", debug_img)

    print(f"🔍 Micro Classifier Locked: {pattern}-inch pattern (Ratio: {ratio:.2f})")
    return pattern

def measure_flattened_length(flat_img, pattern_inches):
    if flat_img is None: return 0, 0

    hsv = cv2.cvtColor(flat_img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

    h, w = mask.shape
    mask[0:int(h*0.80), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    temp_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10 and cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3]]
    if not temp_boxes: return 0, 0

    median_width = np.median([b[2] for b in temp_boxes])
    valid_boxes = [b for b in temp_boxes if (median_width * 0.5) < b[2] < (median_width * 1.5)]
    count = len(valid_boxes)

    display_img = flat_img.copy()
    for bx, by, bw, bh in valid_boxes:
        cv2.rectangle(display_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 4)
    
    cv2.imwrite("proof_3_tape_strip_count.jpg", display_img)
    total_feet = (count * pattern_inches) / 12
    return count, total_feet

def calculate_height_and_area(flat_img, total_length_feet):
    if flat_img is None: return None, None

    pixel_height, pixel_width = flat_img.shape[:2]
    feet_per_pixel = total_length_feet / pixel_width
    trailer_height_feet = pixel_height * feet_per_pixel
    total_area_sqft = total_length_feet * trailer_height_feet

    debug_img = flat_img.copy()
    cv2.line(debug_img, (50, 50), (pixel_width - 50, 50), (0, 255, 255), 5) 
    cv2.line(debug_img, (50, 50), (50, pixel_height - 50), (0, 255, 255), 5) 
    cv2.putText(debug_img, f"L: {total_length_feet:.2f} ft", (int(pixel_width/2)-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
    cv2.putText(debug_img, f"H: {trailer_height_feet:.2f} ft", (70, int(pixel_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
    
    cv2.imwrite("proof_4_dimensions.jpg", debug_img)
    return trailer_height_feet, total_area_sqft

def detect_fleet_brand(flat_img, yolo_logo_model):
    if flat_img is None: return None, None

    results = yolo_logo_model(flat_img, conf=0.5, verbose=False)[0]
    debug_img = flat_img.copy()

    if len(results.boxes) == 0:
        return "UNBRANDED / UNKNOWN", None

    # Grab the first detected logo
    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cls_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    primary_brand = yolo_logo_model.names[cls_id].upper()

    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.putText(debug_img, f"{primary_brand} ({conf:.2f})", (x1, max(40, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.imwrite("proof_5_semantic_brand.jpg", debug_img)

    return primary_brand, [x1, y1, x2, y2]

def calculate_true_ink_area(flat_img, box_coords, total_length_feet):
    if flat_img is None or box_coords is None: return 0

    feet_per_pixel = total_length_feet / flat_img.shape[1]
    sq_ft_per_pixel = feet_per_pixel ** 2  

    x1, y1, x2, y2 = box_coords
    roi = flat_img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels = cv2.countNonZero(binary)
    black_pixels = (binary.shape[0] * binary.shape[1]) - white_pixels

    ink_mask = cv2.bitwise_not(binary) if white_pixels > black_pixels else binary
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    final_ink_pixels = cv2.countNonZero(ink_mask)
    true_ink_sqft = final_ink_pixels * sq_ft_per_pixel

    # Build the side-by-side visual proof
    ink_mask_colored = cv2.cvtColor(ink_mask, cv2.COLOR_GRAY2BGR)
    overlay = roi.copy()
    overlay[ink_mask > 0] = [0, 255, 0]
    combined_proof = np.hstack((roi, ink_mask_colored, overlay))
    cv2.imwrite("proof_6_ink_extraction.jpg", combined_proof)

    return true_ink_sqft


# ==========================================
# MASTER EXECUTION PIPELINE
# ==========================================
print("\n🚀 Initiating Fleet Measurement & Analysis Pipeline...")

flat_canvas = flatten_trailer_side(MACRO_RAW_PATH, trailer_model)
tape_pattern = classify_tape_pattern(MICRO_RAW_PATH)

if flat_canvas is None or tape_pattern is None:
    print("❌ Pipeline failed during early extraction phases. Check input images.")
    sys.exit(1)

final_count, final_length = measure_flattened_length(flat_canvas, tape_pattern)
final_height, final_area = calculate_height_and_area(flat_canvas, final_length)

fleet_brand, logo_coords = detect_fleet_brand(flat_canvas, logo_model)

single_unit_brands = ["AMAZON", "COCACOLA"]
set_of_items_brands = ["FEDEX", "COSTCO", "WALMART"]

manufacturing_type = "Manual Review Required"
if fleet_brand in single_unit_brands:
    manufacturing_type = "Single Unit (Full Wrap)"
elif fleet_brand in set_of_items_brands:
    manufacturing_type = "Set of Items (Discrete Decals)"

true_ink_sqft = 0
if logo_coords is not None:
    print(f"🔍 Analyzing True Pixel Area for {fleet_brand}...")
    true_ink_sqft = calculate_true_ink_area(flat_canvas, logo_coords, final_length)

# --- TERMINAL MANIFEST ---
print("\n" + "█"*55)
print(" 🚚 LOWEN CORP: AUTOMATED PRODUCTION MANIFEST 🚚 ")
print("█"*55)
print(f"🏢 FLEET OWNER          : {fleet_brand}")
print(f"📏 CALCULATED LENGTH    : {final_length:.2f} ft")
print(f"📐 TRAILER HEIGHT       : {final_height:.2f} ft")
print(f"🟩 TOTAL TRAILER AREA   : {final_area:.2f} Sq. Ft.")
print("-" * 55)
print(f"🛠️  MANUFACTURING TYPE   : {manufacturing_type}")

if manufacturing_type == "Single Unit (Full Wrap)":
    print(f"🔵 TOTAL TRAILER WRAP    : {final_area:.2f} Sq. Ft.")
    print(f"✂️  LOGO CUTOUT (WASTE)   : {true_ink_sqft:.2f} Sq. Ft.")
    print(f"✅ REQUIRED GRAPHIC AREA : {(final_area - true_ink_sqft):.2f} Sq. Ft.")
elif manufacturing_type == "Set of Items (Discrete Decals)":
    print(f"💧 BILLABLE VINYL (DECAL): {true_ink_sqft:.2f} Sq. Ft.")

print("█"*55 + "\n")
print("✅ Visual proofs successfully saved to local directory (proof_1 through proof_6).")