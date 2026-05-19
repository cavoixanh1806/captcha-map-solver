import os
import csv
import time
from PIL import Image
from solve_captcha import CaptchaSolver

def load_metadata(metadata_path: str) -> dict:
    """Load ground-truth labels from data/metadata.csv"""
    labels = {}
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata file not found at {metadata_path}")
        return labels
        
    with open(metadata_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2:
                filename, text = row[0], row[1]
                labels[filename] = text.strip().upper()
    return labels

def main():
    checkpoint_path = "best-epoch052.ckpt"
    metadata_path = "data/metadata.csv"
    data_dir = "data"
    
    # 1. Load metadata
    print("[INFO] Loading ground-truth labels from metadata.csv...")
    ground_truth = load_metadata(metadata_path)
    if not ground_truth:
        print("[ERROR] No metadata loaded. Exiting.")
        return
        
    # 2. Get user input for range
    print("\n=== SELECT IMAGE RANGE TO TEST ===")
    try:
        start_val = input("Enter start index (e.g. 0): ").strip()
        start_idx = int(start_val) if start_val != "" else 0
        
        end_val = input("Enter end index (e.g. 20): ").strip()
        end_idx = int(end_val) if end_val != "" else 20
        
        if start_idx < 0 or end_idx < 0:
            print("[ERROR] Index must be non-negative!")
            return
        if start_idx > end_idx:
            print("[ERROR] Start index cannot be greater than end index!")
            return
    except ValueError:
        print("[ERROR] Please enter valid integers!")
        return
        
    # Generate list of filenames within the range
    target_images = []
    for idx in range(start_idx, end_idx + 1):
        filename = f"map_{idx:05d}.png"
        filepath = os.path.join(data_dir, filename)
        
        # Check if the file exists and is in the metadata
        if os.path.exists(filepath):
            if filename in ground_truth:
                target_images.append(filename)
            else:
                print(f"[WARN] File '{filename}' exists but has no label in metadata.csv. Skipping.")
        else:
            print(f"[WARN] File '{filename}' not found at '{filepath}'. Skipping.")
            
    num_samples = len(target_images)
    if num_samples == 0:
        print("[ERROR] No valid images found in the specified range! Exiting.")
        return
        
    print(f"\n[INFO] Found {num_samples} valid images to test within the range {start_idx} to {end_idx}.")
    
    # 3. Initialize Captcha Solver
    solver = CaptchaSolver(checkpoint_path=checkpoint_path)
    
    # 4. Run Batch Inference
    print("\n" + "="*70)
    print(f"{'IMAGE':<20} | {'GROUND TRUTH':<15} | {'PREDICTION':<15} | {'STATUS':<10}")
    print("="*70)
    
    correct_count = 0
    total_time = 0.0
    
    for filename in target_images:
        filepath = os.path.join(data_dir, filename)
        true_label = ground_truth[filename]
        
        # Benchmark single prediction time
        start_time = time.time()
        pred_label = solver.solve(filepath)
        inference_time = time.time() - start_time
        
        total_time += inference_time
        
        is_correct = (pred_label == true_label)
        if is_correct:
            correct_count += 1
            status = "OK"
        else:
            status = "FAILED"
            
        print(f"{filename:<20} | {true_label:<15} | {pred_label:<15} | {status:<10}")
        
    print("="*70)
    
    # 5. Print final statistics
    accuracy = (correct_count / num_samples) * 100
    avg_time = (total_time / num_samples) * 1000  # in milliseconds
    
    print("\n" + "="*40)
    print("           BENCHMARK REPORT")
    print("="*40)
    print(f"Index Range Tested   : {start_idx} to {end_idx}")
    print(f"Total Images Tested  : {num_samples}")
    print(f"Correct Predictions  : {correct_count}")
    print(f"Accuracy Rate        : {accuracy:.1f}%")
    print(f"Total Time Elapsed   : {total_time:.2f} seconds")
    print(f"Average Time/Image   : {avg_time:.1f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
