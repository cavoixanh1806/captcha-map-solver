import os
import sys
import json
import base64
import time
from io import BytesIO
from typing import Optional
from PIL import Image

import asyncio

# Add parent directory to sys.path so we can import modules from TrainAI root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from solve_captcha import CaptchaSolver

# 1. Initialize FastAPI Application
app = FastAPI(
    title="CAPTCHA Solver Microservice",
    description="High-performance TrOCR-based CAPTCHA solving API",
    version="1.0.0"
)

# Enable CORS for cross-origin integration (useful for web dashboards/bots)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global Solver Instance, Lock & Config Loader
solver: Optional[CaptchaSolver] = None
model_lock = asyncio.Lock()
CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")

def load_config() -> dict:
    default_config = {
        "checkpoint_path": "best-epoch032.ckpt",
        "host": "0.0.0.0",
        "port": 5000
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading config.json: {e}. Using defaults.")
    return default_config

config = load_config()

# 3. Lifecycle Events: Load Model on Startup
@app.on_event("startup")
def startup_event():
    global solver
    checkpoint_name = config.get("checkpoint_path", "best-epoch032.ckpt")
    
    # Try looking in parent dir if it's a relative path
    checkpoint_path = checkpoint_name
    if not os.path.isabs(checkpoint_path):
        # First check parent dir (TrainAI root)
        parent_check = os.path.join(PARENT_DIR, checkpoint_name)
        if os.path.exists(parent_check):
            checkpoint_path = parent_check
        else:
            # Second check current dir (api/)
            current_check = os.path.join(CURRENT_DIR, checkpoint_name)
            if os.path.exists(current_check):
                checkpoint_path = current_check

    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"\n[CRITICAL ERROR] Checkpoint file '{checkpoint_name}' not found!")
        print(f"================================================================================")
        print(f"NOTE: Since model checkpoint files (*.ckpt) are ignored by git (due to being ~4GB),")
        print(f"you must manually copy '{checkpoint_name}' from your local machine to this machine")
        print(f"and place it in: {PARENT_DIR}")
        print(f"================================================================================\n")
        solver = None
    else:
        print(f"[BOOT] Initializing model solver with checkpoint: {checkpoint_path}")
        try:
            solver = CaptchaSolver(checkpoint_path=checkpoint_path)
            print("[BOOT] Active model is loaded and ready for predictions!")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load CAPTCHA model: {e}")
            # We don't crash the server so health-check can report the error details
            solver = None

# 4. Request Models
class Base64Request(BaseModel):
    image_base64: str

# 5. Core API Endpoints

@app.get("/health", tags=["System"])
def health_check():
    """Check the health status of the API and verify if the AI model is loaded."""
    if solver is None:
        return {
            "status": "unhealthy",
            "message": "CAPTCHA Solver model failed to load. Please verify checkpoint_path in config.json.",
            "checkpoint": config.get("checkpoint_path")
        }
    return {
        "status": "healthy",
        "message": "CAPTCHA solver model is fully operational in memory.",
        "device": getattr(solver, "device", "unknown"),
        "checkpoint": config.get("checkpoint_path")
    }

@app.post("/solve-file", tags=["Inference"])
async def solve_by_file_upload(file: UploadFile = File(...)):
    """
    Solve a CAPTCHA using an uploaded image file (PNG, JPG, etc.).
    """
    if solver is None:
        raise HTTPException(status_code=503, detail="Model is currently not loaded. Check /health status.")

    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Thread-safe prediction lock for sequential processing
        async with model_lock:
            result = solver.solve(image)
        
        if "[Error]" in result:
            raise HTTPException(status_code=500, detail=result)
            
        elapsed = time.time() - start_time
        return {
            "success": True,
            "captcha": result,
            "inference_time_ms": round(elapsed * 1000, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/solve-base64", tags=["Inference"])
async def solve_by_base64(payload: Base64Request):
    """
    Solve a CAPTCHA by sending a base64-encoded image string.
    
    Format: Accepts either raw base64 or base64 with data URI prefix (e.g. data:image/png;base64,iVBOR...)
    """
    if solver is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health status.")

    start_time = time.time()
    try:
        # Clean up data URI scheme if present
        b64_data = payload.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]

        # Decode base64 bytes
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Thread-safe prediction lock for sequential processing
        async with model_lock:
            result = solver.solve(image)
        
        if "[Error]" in result:
            raise HTTPException(status_code=500, detail=result)
            
        elapsed = time.time() - start_time
        return {
            "success": True,
            "captcha": result,
            "inference_time_ms": round(elapsed * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted Base64 image payload: {str(e)}")

# 6. Bootstrapping Script
if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting CAPTCHA Solver API on http://{config['host']}:{config['port']}")
    print(f"Interactive Swagger Documentation available at: http://{config['host']}:{config['port']}/docs")
    
    uvicorn.run(
        "main:app", 
        host=config["host"], 
        port=config["port"], 
        reload=False  # Keep false in production/cuda to prevent dual-memory allocations
    )
