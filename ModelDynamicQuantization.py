import torch
from ultralytics import YOLO

base_model = YOLO("yolov8n-face.pt").model
base_model.eval()

quantized_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype = torch.qint8)

torch.save(quantized_model.state_dict(), "Model01-dynamic.pt")