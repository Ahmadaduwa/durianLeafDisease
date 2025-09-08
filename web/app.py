import os
import time
import io
import logging
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, defaultdict
import threading
import psutil

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError as e:
    GRADCAM_AVAILABLE = False
    print("⚠️ ไม่สามารถนำเข้า pytorch-grad-cam: %s", str(e))
    print("กรุณาติดตั้งด้วย: pip install pytorch-grad-cam")

# -------------------- User settings --------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "./../transferLearning_aug/resnet50/resnet50_checkpoint_fold4.pt")
CLASS_NAMES: List[str] = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT"
]
IMAGE_SIZE = 224
TOPK = 5
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USE_GPU = torch.cuda.is_available()

# -------------------- Setup logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('durian_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Helper functions --------------------
def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def build_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_model(model_path: str, device):
    if not os.path.exists(model_path):
        error_msg = f"ไม่พบไฟล์โมเดล: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model = resnet50(num_classes=len(CLASS_NAMES))
            state_dict = checkpoint['state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logger.info(f"โหลด state_dict จาก {model_path}")
        else:
            model = checkpoint
            if hasattr(model, "module"):
                model = model.module
            logger.info(f"โหลดโมเดลเต็มจาก {model_path}")
        
        if not isinstance(model, nn.Module):
            error_msg = "ไฟล์โมเดลไม่ใช่ PyTorch model หรือ state_dict"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        model.to(device)
        model.eval()
        logger.info(f"โมเดลพร้อมใช้งานบน {device}")
        return model
    except Exception as e:
        error_msg = f"ไม่สามารถโหลดโมเดล: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def find_last_conv_layer(model):
    try:
        # ใช้ layer4.2.conv3 ตามที่ระบุ
        last_conv = model.layer4[-1].conv3
        last_name = "layer4.2.conv3"
        logger.info(f"Grad-CAM target layer: {last_name}")
        return last_conv
    except AttributeError:
        logger.warning("ไม่สามารถใช้ layer4.2.conv3 ได้ ลองค้นหา Conv2d layer อื่น")
        last_conv = None
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_name = name
        if last_conv is None:
            raise ValueError("ไม่พบ Conv2d layer ในโมเดล")
        logger.info(f"Grad-CAM target layer (fallback): {last_name}")
        return last_conv

def read_image_as_float(img_pil: Image.Image, size: tuple) -> np.ndarray:
    img = img_pil.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def numpy_to_base64(img_array: np.ndarray) -> str:
    try:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.info(f"Generated base64 image, size: {len(img_base64)} bytes")
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Failed to convert numpy array to base64: {e}")
        return None

# -------------------- Performance tracking --------------------
class PerformanceTracker:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.predictions = deque(maxlen=max_history)
        self.system_metrics = deque(maxlen=max_history)
        self.error_logs = deque(maxlen=100)
        self.class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        self.lock = threading.Lock()
        
    def log_prediction(self, latency_ms: float, confidence: float, predicted_class: str, 
                      image_size: tuple, success: bool = True, error_msg: str = None, 
                      gradcam_time: float = 0):
        with self.lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "confidence": confidence,
                "predicted_class": predicted_class,
                "image_size": image_size,
                "success": success,
                "error_msg": error_msg,
                "gradcam_time_ms": gradcam_time
            }
            self.predictions.append(log_entry)
            
            if not success and error_msg:
                self.error_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "type": "prediction_error"
                })
    
    def log_system_metrics(self):
        with self.lock:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                self.system_metrics.append({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3)
                })
            except:
                pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        with self.lock:
            if not self.predictions:
                return {"status": "no_data"}
            
            successful_predictions = [p for p in self.predictions if p["success"]]
            
            if not successful_predictions:
                return {"status": "no_successful_predictions"}
            
            latencies = [p["latency_ms"] for p in successful_predictions]
            confidences = [p["confidence"] for p in successful_predictions]
            gradcam_times = [p.get("gradcam_time_ms", 0) for p in successful_predictions]
            
            class_counts = defaultdict(int)
            for p in successful_predictions:
                class_counts[p["predicted_class"]] += 1
            
            return {
                "status": "ok",
                "total_predictions": len(self.predictions),
                "successful_predictions": len(successful_predictions),
                "error_rate": (len(self.predictions) - len(successful_predictions)) / len(self.predictions) * 100,
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "avg_confidence": sum(confidences) / len(confidences),
                "avg_gradcam_time_ms": sum(gradcam_times) / len(gradcam_times) if gradcam_times else 0,
                "class_distribution": dict(class_counts),
                "recent_errors": list(self.error_logs)[-5:],
                "system_metrics": list(self.system_metrics)[-10:] if self.system_metrics else []
            }

# -------------------- Initialize app --------------------
app = Flask(__name__, static_folder=".", static_url_path="")
performance_tracker = PerformanceTracker()

def monitor_system():
    while True:
        performance_tracker.log_system_metrics()
        time.sleep(5)

monitor_thread = threading.Thread(target=monitor_system, daemon=True)
monitor_thread.start()

# -------------------- Load model and preprocess --------------------
device = torch.device("cuda" if USE_GPU else "cpu")
model = None
preprocess = None
target_layer = None

try:
    if not GRADCAM_AVAILABLE:
        raise ImportError("pytorch-grad-cam ไม่ได้ติดตั้ง กรุณารัน: pip install pytorch-grad-cam")
    model = load_model(MODEL_PATH, device)
    preprocess = build_preprocess(IMAGE_SIZE)
    target_layer = find_last_conv_layer(model)
    logger.info("🚀 ระบบพร้อมใช้งาน!")
except Exception as e:
    error_msg = f"❌ ข้อผิดพลาดขณะโหลดโมเดลหรือตั้งค่า Grad-CAM: {e}"
    logger.error(error_msg, exc_info=True)
    performance_tracker.error_logs.append({
        "timestamp": datetime.now().isoformat(),
        "error": str(e),
        "type": "initialization_error"
    })

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/performance", methods=["GET"])
def get_performance_stats():
    stats = performance_tracker.get_performance_stats()
    return jsonify(stats)

@app.route("/debug", methods=["GET"])
def debug():
    layer_info = None
    if model and target_layer:
        layer_info = {
            "target_layer": str(target_layer),
            "is_conv2d": isinstance(target_layer, nn.Conv2d),
            "model_layers": [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
        }
    return jsonify({
        "model_loaded": model is not None,
        "device": str(device),
        "gradcam_initialized": target_layer is not None,
        "gradcam_available": GRADCAM_AVAILABLE,
        "class_names": CLASS_NAMES,
        "layer_info": layer_info,
        "model_structure": str(model) if model else "None",
        "model_path_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False
    })

@app.route("/predict", methods=["POST"])
def predict():
    global model, device, preprocess, target_layer, CLASS_NAMES
    
    start_time = time.time()
    predicted_class = "unknown"
    confidence = 0.0
    image_size = (0, 0)
    gradcam_time = 0.0
    
    try:
        if model is None or preprocess is None or target_layer is None:
            error_msg = "โมเดลหรือการตั้งค่า Grad-CAM ยังไม่ถูกโหลด"
            if not GRADCAM_AVAILABLE:
                error_msg += " (pytorch-grad-cam ไม่ได้ติดตั้ง)"
            logger.error(f"🔴 {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", (0, 0), False, error_msg
            )
            return jsonify({"error": error_msg}), 500

        if "file" not in request.files:
            error_msg = "ไม่พบไฟล์ในคำขอ (ต้องใช้ key ชื่อ 'file')"
            logger.warning(f"⚠️ {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", (0, 0), False, error_msg
            )
            return jsonify({"error": error_msg}), 400

        f = request.files["file"]
        if f.filename == "":
            error_msg = "ยังไม่ได้เลือกไฟล์"
            logger.warning(f"⚠️ {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", (0, 0), False, error_msg
            )
            return jsonify({"error": error_msg}), 400

        if not allowed_file(f.filename):
            error_msg = f"นามสกุลไฟล์ไม่รองรับ: {f.filename}"
            logger.warning(f"⚠️ {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", (0, 0), False, error_msg
            )
            return jsonify({"error": "นามสกุลไฟล์ไม่รองรับ"}), 400

        try:
            img_bytes = f.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_size = img_pil.size
            img_np = read_image_as_float(img_pil, (IMAGE_SIZE, IMAGE_SIZE))
            logger.info(f"📸 ได้รับภาพ: {f.filename} ({image_size[0]}x{image_size[1]})")
        except Exception as e:
            error_msg = f"เปิดไฟล์รูปภาพไม่สำเร็จ: {e}"
            logger.error(f"🔴 {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", image_size, False, error_msg
            )
            return jsonify({"error": error_msg}), 400

        preprocess_start = time.time()
        x = preprocess(img_pil).unsqueeze(0).to(device)
        preprocess_time = time.time() - preprocess_start

        inference_start = time.time()
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            probs_topk, idx_topk = torch.topk(probs, k=min(TOPK, probs.size(1)), dim=1)
        inference_time = time.time() - inference_start

        gradcam_img = None
        gradcam_data = None
        gradcam_stats = None
        gradcam_time = 0.0
        if target_layer is not None and request.form.get('enable_gradcam', 'true').lower() == 'true':
            try:
                gradcam_start = time.time()
                predicted_idx = int(idx_topk[0][0].item())
                target = ClassifierOutputTarget(predicted_idx)
                cam = GradCAM(model=model, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=x, targets=[target])
                mask = grayscale_cam[0]
                
                gradcam_img = show_cam_on_image(img_np, mask, use_rgb=True)
                logger.info(f"Grad-CAM image shape: {gradcam_img.shape}")
                gradcam_data = numpy_to_base64(gradcam_img)
                if gradcam_data is None:
                    raise ValueError("Failed to convert Grad-CAM image to base64")
                gradcam_time = (time.time() - gradcam_start) * 1000
                
                gradcam_stats = {
                    "max": float(mask.max()),
                    "min": float(mask.min()),
                    "mean": float(mask.mean()),
                    "shape": list(mask.shape)
                }
                
                if mask.max() <= 0:
                    logger.warning("Grad-CAM: CAM is all zeros")
                    gradcam_stats["warning"] = "CAM is all zeros"
                
                logger.info(f"🔍 Grad-CAM generated: {gradcam_time:.1f}ms, stats: {gradcam_stats}")
            except Exception as e:
                logger.error(f"⚠️ Grad-CAM failed: {e}", exc_info=True)
                gradcam_time = 0
                gradcam_stats = {"error": str(e)}

        total_latency_ms = (time.time() - start_time) * 1000

        idxs = idx_topk.squeeze(0).cpu().tolist()
        probs_vals = probs_topk.squeeze(0).cpu().tolist()

        results = []
        for idx, p in zip(idxs, probs_vals):
            label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"class_{idx}"
            results.append({"index": int(idx), "label": label, "prob": float(p)})

        predicted_class = results[0]["label"] if results else "unknown"
        confidence = results[0]["prob"] if results else 0.0
        
        performance_tracker.log_prediction(
            total_latency_ms, confidence, predicted_class, image_size, True, 
            gradcam_time=gradcam_time
        )
        
        logger.info(f"✅ Prediction: {predicted_class} ({confidence:.2%}) - {total_latency_ms:.1f}ms")

        response = {
            "topk": results,
            "latency_ms": int(total_latency_ms),
            "preprocessing_time_ms": int(preprocess_time * 1000),
            "inference_time_ms": int(inference_time * 1000),
            "gradcam_time_ms": int(gradcam_time),
            "cam_stats": gradcam_stats
        }
        
        if gradcam_data:
            response["gradcam_image"] = gradcam_data
            
        return jsonify(response)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"🔴 {error_msg}", exc_info=True)
        performance_tracker.log_prediction(
            (time.time() - start_time) * 1000, 0.0, predicted_class, image_size, False, error_msg,
            gradcam_time=gradcam_time
        )
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("🌟 เริ่มต้นเซิร์ฟเวอร์ทำนายโรคใบทุเรียน (พร้อม Grad-CAM)")
    app.run(host="0.0.0.0", port=5000, debug=True)