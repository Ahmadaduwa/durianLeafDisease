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
# -------------------------------------------------------

class GradCAM:
    """Grad-CAM implementation for model interpretation"""
    
    def __init__(self, model, target_layer_name: str = 'layer4'):
        self.model = model
        self.target_layer = self._get_target_layer(target_layer_name)
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _get_target_layer(self, layer_name: str):
        """Get the target layer from model"""
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        else:
            # For ResNet models, try common layer names
            layer_mapping = {
                'layer4': 'layer4',
                'layer3': 'layer3', 
                'layer2': 'layer2',
                'layer1': 'layer1'
            }
            if layer_name in layer_mapping and hasattr(self.model, layer_mapping[layer_name]):
                return getattr(self.model, layer_mapping[layer_name])
        
        # Fallback: find last convolutional layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                logger.info(f"Using layer '{name}' for Grad-CAM")
                return module
        
        raise ValueError("Could not find suitable layer for Grad-CAM")
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate CAM heatmap"""
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().numpy()  # [C, H, W]
        activations = self.activations[0].cpu().numpy()  # [C, H, W]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU to the heatmap
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_cam(self, input_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Create visualization by overlaying CAM on input image"""
        # Resize CAM to match input image size
        h, w = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure input_image is in the right format
        if input_image.dtype != np.uint8:
            input_image = (input_image * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        superimposed = heatmap * alpha + input_image * (1 - alpha)
        superimposed = superimposed.astype(np.uint8)
        
        return superimposed

# Performance tracking
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
            
            # Calculate class distribution
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
                "recent_errors": list(self.error_logs)[-5:],  # Last 5 errors
                "system_metrics": list(self.system_metrics)[-10:] if self.system_metrics else []
            }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('durian_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
performance_tracker = PerformanceTracker()

# System monitoring thread
def monitor_system():
    while True:
        performance_tracker.log_system_metrics()
        time.sleep(5)  # Log every 5 seconds

monitor_thread = threading.Thread(target=monitor_system, daemon=True)
monitor_thread.start()

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def build_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        error_msg = f"ไม่พบไฟล์โมเดล: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        loaded = torch.load(model_path, map_location=device)
        logger.info(f"โหลดไฟล์โมเดลสำเร็จ: {model_path}")
    except Exception as e:
        error_msg = f"ไม่สามารถอ่านไฟล์โมเดลได้: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if isinstance(loaded, dict) and any(k in loaded for k in ["state_dict", "model_state_dict"]):
        error_msg = "ไฟล์เป็น checkpoint (state_dict) — ต้องมีสคริปต์สร้างสถาปัตยกรรมเดิมแล้ว load_state_dict()"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    model = loaded
    if not hasattr(model, "eval"):
        error_msg = "รูปแบบไฟล์โมเดลไม่ถูกต้อง (ไม่ใช่โมเดล PyTorch ที่บันทึกทั้งหมด)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    model.to(device)
    model.eval()
    logger.info(f"โมเดลพร้อมใช้งานบนอุปกรณ์: {device}")
    return model, device

def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 string for web display"""
    # Convert to PIL Image
    img_pil = Image.fromarray(img_array)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# โหลดโมเดลและ preprocess ตอนเริ่มแอป
try:
    model, device = load_model(MODEL_PATH)
    preprocess = build_preprocess(IMAGE_SIZE)
    gradcam = GradCAM(model)
    logger.info("🚀 ระบบพร้อมใช้งาน (รวม Grad-CAM)!")
except Exception as e:
    model = None
    device = torch.device("cpu")
    preprocess = build_preprocess(IMAGE_SIZE)
    gradcam = None
    error_msg = f"❌ ข้อผิดพลาดขณะโหลดโมเดล: {e}"
    logger.error(error_msg)
    performance_tracker.error_logs.append({
        "timestamp": datetime.now().isoformat(),
        "error": str(e),
        "type": "model_loading_error"
    })

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/performance", methods=["GET"])
def get_performance_stats():
    """API endpoint สำหรับดู performance statistics"""
    stats = performance_tracker.get_performance_stats()
    return jsonify(stats)

@app.route("/predict", methods=["POST"])
def predict():
    global model, device, preprocess, gradcam, CLASS_NAMES
    
    start_time = time.time()
    predicted_class = "unknown"
    confidence = 0.0
    image_size = (0, 0)
    gradcam_time = 0.0
    
    try:
        if model is None:
            error_msg = "โมเดลยังไม่ถูกโหลดบนเซิร์ฟเวอร์"
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

        # อ่านภาพเข้าหน่วยความจำ
        try:
            img_bytes = f.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            image_size = img_pil.size
            
            # Convert to numpy for Grad-CAM visualization
            img_np = np.array(img_pil)
            
            logger.info(f"📸 ได้รับภาพ: {f.filename} ({image_size[0]}x{image_size[1]})")
        except Exception as e:
            error_msg = f"เปิดไฟล์รูปภาพไม่สำเร็จ: {e}"
            logger.error(f"🔴 {error_msg}")
            performance_tracker.log_prediction(
                (time.time() - start_time) * 1000, 0.0, "none", image_size, False, error_msg
            )
            return jsonify({"error": error_msg}), 400

        # พรีโพรเซส และทำนาย
        x = preprocess(img_pil).unsqueeze(0).to(device)
        t0 = time.time()
        
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            probs_topk, idx_topk = torch.topk(probs, k=min(TOPK, probs.size(1)), dim=1)
        
        inference_time = time.time() - t0

        # Generate Grad-CAM
        gradcam_img = None
        gradcam_data = None
        if gradcam is not None and request.form.get('enable_gradcam', 'true').lower() == 'true':
            try:
                gradcam_start = time.time()
                
                # Get predicted class index
                predicted_idx = idx_topk[0][0].item()
                
                # Generate CAM
                cam = gradcam.generate_cam(x, predicted_idx)
                
                # Create visualization
                gradcam_img = gradcam.visualize_cam(img_np, cam, alpha=0.4)
                gradcam_data = numpy_to_base64(gradcam_img)
                
                gradcam_time = (time.time() - gradcam_start) * 1000
                logger.info(f"🔍 Grad-CAM สร้างสำเร็จ: {gradcam_time:.1f}ms")
                
            except Exception as e:
                logger.warning(f"⚠️ Grad-CAM ล้มเหลว: {e}")
                gradcam_time = 0

        total_latency_ms = (time.time() - start_time) * 1000

        idxs = idx_topk.squeeze(0).cpu().tolist()
        probs_vals = probs_topk.squeeze(0).cpu().tolist()

        results = []
        for idx, p in zip(idxs, probs_vals):
            label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"class_{idx}"
            results.append({"index": int(idx), "label": label, "prob": float(p)})

        # Log successful prediction
        predicted_class = results[0]["label"] if results else "unknown"
        confidence = results[0]["prob"] if results else 0.0
        
        performance_tracker.log_prediction(
            total_latency_ms, confidence, predicted_class, image_size, True, 
            gradcam_time=gradcam_time
        )
        
        logger.info(f"✅ ทำนายสำเร็จ: {predicted_class} ({confidence:.2%}) - {total_latency_ms:.1f}ms")

        response = {
            "topk": results, 
            "latency_ms": int(total_latency_ms),
            "inference_time_ms": int(inference_time * 1000),
            "preprocessing_time_ms": int((t0 - start_time) * 1000),
            "gradcam_time_ms": int(gradcam_time)
        }
        
        if gradcam_data:
            response["gradcam_image"] = gradcam_data
            
        return jsonify(response)

    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาดไม่คาดคิด: {e}"
        logger.error(f"🔴 {error_msg}")
        performance_tracker.log_prediction(
            (time.time() - start_time) * 1000, 0.0, predicted_class, image_size, False, error_msg,
            gradcam_time=gradcam_time
        )
        return jsonify({"error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

if __name__ == "__main__":
    logger.info("🌟 เริ่มต้นเซิร์ฟเวอร์ทำนายโรคใบทุเรียน (พร้อม Grad-CAM)")
    app.run(host="0.0.0.0", port=5000, debug=True)