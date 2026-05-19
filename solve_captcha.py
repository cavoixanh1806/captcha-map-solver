import os
import torch
from PIL import Image
from src.trocr import TrOCRLitModel

class CaptchaSolver:
    def __init__(self, checkpoint_path: str = "best-epoch052.ckpt"):
        """Initialize and load the TrOCR model into memory (do this once for fast inference)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading model from {checkpoint_path} onto device: {self.device}...")
        
        # Load model
        self.model = TrOCRLitModel.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)
        print("[SUCCESS] Model loaded successfully!")

    def solve(self, image_input) -> str:
        """Solve a CAPTCHA image given a file path or PIL Image object.
        
        Args:
            image_input: Either a file path string (str) or a PIL.Image.Image object.
            
        Returns:
            The predicted 5-character uppercase CAPTCHA string.
        """
        try:
            # If input is a file path
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return f"[Error] File not found: {image_input}"
                img = Image.open(image_input).convert("RGB")
            else:
                # If input is already a PIL Image
                img = image_input.convert("RGB")

            # Preprocess image
            pixel_values = self.model.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
            
            # Inference
            with torch.no_grad():
                gen = self.model.model.generate(pixel_values)
                
            # Decode predictions
            pred_text = self.model.processor.batch_decode(gen, skip_special_tokens=True)[0]
            
            # Format output (remove spaces, uppercase)
            return pred_text.replace(" ", "").upper()
            
        except Exception as e:
            return f"[Error] Inference error: {str(e)}"

# Example usage when running directly from CMD/PowerShell
if __name__ == "__main__":
    import sys
    
    ckpt = "best-epoch052.ckpt"
    test_image = "data/map_00000.png"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
    if not os.path.exists(ckpt):
        print(f"[ERROR] Checkpoint file {ckpt} not found in the root directory!")
        sys.exit(1)
        
    # Initialize solver
    solver = CaptchaSolver(checkpoint_path=ckpt)
    
    # Run test prediction
    result = solver.solve(test_image)
    print(f"\n[RESULT] {test_image} => {result}")
