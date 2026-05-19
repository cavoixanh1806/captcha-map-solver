import os
import re
import time
import glob
import argparse
import urllib.request
import urllib.error
import json

def parse_label(filename: str) -> str:
    """
    Extract the ground truth label from the CAPTCHA image filename.
    Format support:
      - map_33JJP.png -> 33JJP
      - map_33JJP_1779127465638.png -> 33JJP
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    if name_without_ext.startswith("map_"):
        parts = name_without_ext.split("_")
        if len(parts) >= 2:
            return parts[1]
    
    # Fallback: find any 5-character uppercase alphanumeric block in the filename
    match = re.search(r"[A-Z0-9]{5}", basename)
    if match:
        return match.group(0)
        
    return "UNKNOWN"

def solve_captcha_via_api(url: str, filepath: str) -> dict:
    """
    Sends the image file to the FastAPI solve-file endpoint using standard urllib.
    This avoids external dependencies like 'requests' if it is not in the system.
    """
    try:
        with open(filepath, "rb") as f:
            file_bytes = f.read()
            
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        # Build multipart/form-data payload
        body = []
        body.append(f"--{boundary}".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(filepath)}"'.encode("utf-8"))
        body.append(b"Content-Type: image/png")
        body.append(b"")
        body.append(file_bytes)
        body.append(f"--{boundary}--".encode("utf-8"))
        body.append(b"")
        
        payload = b"\r\n".join(body)
        
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload))
        }
        
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return {
                "success": res_data.get("success", False),
                "predicted": res_data.get("captcha", "ERROR"),
                "time_ms": res_data.get("inference_time_ms", 0.0),
                "error": None
            }
            
    except urllib.error.URLError as e:
        return {
            "success": False,
            "predicted": "API_DOWN",
            "time_ms": 0.0,
            "error": f"Failed to connect to API: {e.reason if hasattr(e, 'reason') else e}"
        }
    except Exception as e:
        return {
            "success": False,
            "predicted": "ERROR",
            "time_ms": 0.0,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate CAPTCHA Solver API on image files.")
    parser.add_argument("--dir", default="ac", help="Directory containing evaluation images (default: ac)")
    parser.add_argument("--url", default="http://127.0.0.1:5000/solve-file", help="API solve endpoint")
    parser.add_argument("--limit", type=int, default=8, help="Limit number of images to evaluate")
    args = parser.parse_args()
    
    print("==========================================================")
    print("=== CAPTCHA Solver API Evaluation Tool ===")
    print("==========================================================")
    
    # 1. Look for images in the target directory
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(args.dir, pattern)))
        
    # Deduplicate paths (Windows is case-insensitive, glob can return duplicates)
    seen = set()
    image_paths = [p for p in image_paths if not (p.lower() in seen or seen.add(p.lower()))]
    
    # Sort files to ensure stable order
    image_paths.sort()
    
    # Fallback to 'data' directory if 'ac' is empty
    if not image_paths:
        print(f"[WARN] No images found in '{args.dir}/' directory.")
        fallback_dir = "data"
        print(f"Falling back to '{fallback_dir}/' directory for evaluation...")
        for pattern in image_patterns:
            image_paths.extend(glob.glob(os.path.join(fallback_dir, pattern)))
            
        seen = set()
        image_paths = [p for p in image_paths if not (p.lower() in seen or seen.add(p.lower()))]
        image_paths = [p for p in image_paths if not p.endswith("metadata.csv")]
        image_paths.sort()
        
    if not image_paths:
        print("[CRITICAL ERROR] No evaluation images found in either 'ac/' or 'data/'!")
        return
        
    # Limit to 8 images
    eval_set = image_paths[:args.limit]
    print(f"Found total {len(image_paths)} images. Evaluating first {len(eval_set)} images...")
    
    print("\nConnecting to API...")
    # Health check
    health_url = args.url.replace("/solve-file", "/health")
    try:
        with urllib.request.urlopen(health_url, timeout=3) as h_res:
            h_data = json.loads(h_res.read().decode("utf-8"))
            print(f"[OK] Connected to API. Status: {h_data.get('status')} | Active Checkpoint: {h_data.get('checkpoint')}")
    except Exception:
        print(f"[WARNING] API health-check at {health_url} failed. Is the API server running?")
        print("Continuing with predictions anyway...\n")

    # Table Header
    print(f"{'#':<3} | {'File Name':<35} | {'True Label':<10} | {'Pred Label':<10} | {'Status':<10} | {'Time (ms)':<10}")
    print("-" * 90)
    
    correct_count = 0
    total_time = 0.0
    
    for idx, filepath in enumerate(eval_set, 1):
        filename = os.path.basename(filepath)
        true_label = parse_label(filename)
        
        # Call API
        res = solve_captcha_via_api(args.url, filepath)
        
        pred_label = res["predicted"]
        time_ms = res["time_ms"]
        
        status = "❌ WRONG"
        if res["success"] and pred_label.upper() == true_label.upper():
            status = "✅ OK"
            correct_count += 1
            
        if res["error"]:
            status = "⚠️ ERROR"
            pred_label = "FAIL"
            
        total_time += time_ms
        
        # Truncate filename if it's too long for table layout
        disp_name = filename if len(filename) <= 35 else filename[:32] + "..."
        
        err_msg = f" ({res['error']})" if res["error"] else ""
        print(f"{idx:<3} | {disp_name:<35} | {true_label:<10} | {pred_label:<10} | {status:<10}{err_msg} | {time_ms:<10.2f}")
        
    # Print Summary Results
    accuracy = (correct_count / len(eval_set)) * 100
    avg_time = total_time / len(eval_set) if eval_set else 0
    
    print("-" * 90)
    print("=== SUMMARY STATISTICS ===")
    print(f"Total Evaluated : {len(eval_set)}")
    print(f"Correct Solves  : {correct_count}")
    print(f"Exact Accuracy  : {accuracy:.2f}%")
    print(f"Average Speed   : {avg_time:.2f} ms / image")
    print("==========================================================")

if __name__ == "__main__":
    main()
