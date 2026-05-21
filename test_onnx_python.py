import os
import time
import csv
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import TrOCRProcessor
from PIL import Image

def main():
    print("[INFO] Loading ONNX model...")
    processor = TrOCRProcessor.from_pretrained("hf_trocr_model")
    model = ORTModelForVision2Seq.from_pretrained("onnx_model", use_io_binding=False, provider="CPUExecutionProvider")

    print("[INFO] Loading metadata...")
    metadata = {}
    with open("data/metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["filename"]] = row["text"]

    filenames = sorted(list(metadata.keys()))[:500]
    print(f"[INFO] Evaluating {len(filenames)} images...")

    correct = 0
    errors = []

    start_time = time.time()
    for i, file in enumerate(filenames):
        truth = metadata[file]
        img_path = os.path.join("data", file)
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Generate
            generated_ids = model.generate(pixel_values, max_length=10, use_cache=False)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred = pred.replace(" ", "").upper()
            
            if pred == truth:
                correct += 1
            else:
                errors.append((file, truth, pred))
                
        except Exception as e:
            print(f"Error on {file}: {e}")
            errors.append((file, truth, "ERROR"))
            
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(filenames)} - Acc: {correct/(i+1)*100:.2f}%", flush=True)

    eval_time = time.time() - start_time
    accuracy = correct / len(filenames) * 100
    
    print("\n======================================")
    print(f"Evaluation finished in {eval_time:.2f}s ({eval_time/len(filenames)*1000:.2f}ms per image)")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(filenames)})")
    print("Errors:")
    for f, t, p in errors:
        print(f"  {f}: Truth={t}, Pred={p}")
    print("======================================\n")

if __name__ == "__main__":
    main()