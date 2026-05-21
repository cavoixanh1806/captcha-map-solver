import os
import sys
import csv
import time
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
from src.compare_models import load_trocr, predict_one

def main():
    ckpt1_path = os.path.join(PROJECT_ROOT, "best-epoch032.ckpt")
    ckpt2_path = os.path.join(PROJECT_ROOT, "best-epoch072.ckpt")
    
    if not os.path.exists(ckpt1_path):
        print(f"Error: {ckpt1_path} not found.")
        return
    if not os.path.exists(ckpt2_path):
        print(f"Error: {ckpt2_path} not found.")
        return
        
    csv_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load metadata labels
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["text"].strip().upper()

    # Get first 500 images that actually exist
    image_list = []
    for filename, true_label in labels.items():
        img_path = os.path.join(PROJECT_ROOT, "data", filename)
        if os.path.exists(img_path):
            image_list.append((filename, img_path, true_label))
            if len(image_list) >= 500:
                break
                
    if len(image_list) < 500:
        print(f"Warning: Only found {len(image_list)} images (less than 500).")
    else:
        print(f"Testing on first {len(image_list)} images from data directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Eval Model 1
    print(f"\n[1/2] Loading Model 1: best-epoch032.ckpt ...")
    model1 = load_trocr(ckpt1_path, device)
    preds1 = {}
    print("Running inference for Model 1...")
    t0 = time.time()
    for filename, img_path, _ in image_list:
        preds1[filename] = predict_one(model1, img_path, device)
    print(f"Model 1 inference done in {time.time()-t0:.1f}s.")
    del model1
    if device == "cuda":
        torch.cuda.empty_cache()

    # Eval Model 2
    print(f"\n[2/2] Loading Model 2: best-epoch072.ckpt ...")
    model2 = load_trocr(ckpt2_path, device)
    preds2 = {}
    print("Running inference for Model 2...")
    t0 = time.time()
    for filename, img_path, _ in image_list:
        preds2[filename] = predict_one(model2, img_path, device)
    print(f"Model 2 inference done in {time.time()-t0:.1f}s.")
    del model2
    if device == "cuda":
        torch.cuda.empty_cache()

    # Compare and print only errors
    print("\n" + "="*95)
    print(f"{'STT':<5} | {'Tên File':<20} | {'Thật':<8} | {'Model 32':<12} | {'Model 72':<12} | {'Trạng thái'}")
    print("="*95)
    
    wrong_count = 0
    correct1_count = 0
    correct2_count = 0
    stt = 1
    
    for filename, img_path, true_label in image_list:
        pred1 = preds1[filename]
        pred2 = preds2[filename]
        
        ok1 = (pred1 == true_label)
        ok2 = (pred2 == true_label)
        
        if ok1:
            correct1_count += 1
        if ok2:
            correct2_count += 1
            
        if not ok1 or not ok2:
            wrong_count += 1
            if ok1 and not ok2:
                status = "M32 Đúng, M72 Sai"
            elif not ok1 and ok2:
                status = "M32 Sai, M72 Đúng"
            else:
                status = "Cả hai đều Sai"
                
            print(f"{stt:<5} | {filename:<20} | {true_label:<8} | {pred1:<12} | {pred2:<12} | {status}")
            stt += 1
            
    print("="*95)
    total_imgs = len(image_list)
    print(f"Tổng số ảnh test: {total_imgs}")
    print(f"Model 32 chính xác: {correct1_count}/{total_imgs} ({correct1_count/total_imgs*100:.2f}%)")
    print(f"Model 72 chính xác: {correct2_count}/{total_imgs} ({correct2_count/total_imgs*100:.2f}%)")
    print(f"Số lượng ảnh dự đoán sai ở ít nhất một mô hình: {wrong_count}")
    print("="*95)

if __name__ == "__main__":
    main()
