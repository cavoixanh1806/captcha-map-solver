import os
import shutil
from tqdm import tqdm
from solve_captcha import CaptchaSolver

def main():
    checkpoint_path = "best-epoch032.ckpt"
    source_dir = "ac"
    dest_dir = "checker"
    
    # 1. Check directories
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory '{source_dir}' does not exist!")
        return
        
    os.makedirs(dest_dir, exist_ok=True)
    
    # 2. Scan source directory for map_*.png files (sorted chronologically by index)
    raw_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith('.png')])
    if not raw_files:
        print(f"[WARN] No PNG files found in source directory '{source_dir}'!")
        return
        
    print(f"[INFO] Found {len(raw_files)} raw images in '{source_dir}' directory.")
    
    # 3. Load the local AI model
    solver = CaptchaSolver(checkpoint_path=checkpoint_path)
    
    print("\n[INFO] Starting auto-labeling process...")
    success_count = 0
    error_count = 0
    
    # Using tqdm for a beautiful local progress bar
    for filename in tqdm(raw_files, desc="Auto Labeling", unit="img"):
        src_path = os.path.join(source_dir, filename)
        
        # Run AI inference to predict label
        predicted_label = solver.solve(src_path)
        
        # Check for errors in inference
        if "[Error]" in predicted_label or predicted_label == "":
            print(f"\n[WARN] Failed to predict '{filename}': {predicted_label}")
            error_count += 1
            continue
            
        # Target filename: map_[LABEL].png
        dest_filename = f"map_{predicted_label}.png"
        dest_path = os.path.join(dest_dir, dest_filename)
        
        try:
            # Move file to checker directory with the predicted name (deletes from ac)
            shutil.move(src_path, dest_path)
            
            # Extract index from original filename (e.g. map_00123.png -> 123)
            # This ensures each file has a unique, sequentially incremented timestamp
            index_str = filename.lower().replace("map_", "").replace(".png", "")
            index_val = int(index_str) if index_str.isdigit() else 0
            
            # 1779000000 corresponds to a timestamp in 2026.
            # We offset it by index_val seconds so they sort perfectly chronologically.
            mtime = 1779000000 + index_val
            os.utime(dest_path, (mtime, mtime))
            
            success_count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to copy '{filename}': {str(e)}")
            error_count += 1
            
    print("\n" + "="*50)
    print("            AUTO-LABELING COMPLETED")
    print("="*50)
    print(f"Total processed files   : {len(raw_files)}")
    print(f"Successfully labeled    : {success_count} files")
    print(f"Failed/Skipped files    : {error_count} files")
    print(f"Output directory        : {os.path.abspath(dest_dir)}")
    print("="*50)
    print("\n[TIP] All labeled images are now in the 'checker/' directory.")
    print("      You can open your labeling tool to review and verify them!")

if __name__ == "__main__":
    main()
