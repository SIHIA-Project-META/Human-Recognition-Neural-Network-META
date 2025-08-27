import os
from typing import List, Dict, Union, Iterable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO


ArrayLike = Union[np.ndarray, torch.Tensor, Image.Image]
Det = Dict[str, Union[torch.Tensor, int, float]]
FrameDetections = List[Det]


def auto_bgr_to_rgb(img: np.ndarray, force_assume_bgr=False) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 image.")

    if force_assume_bgr:
        return img[..., ::-1].copy()

    blue_mean = img[..., 0].mean()
    red_mean = img[..., 2].mean()

    if blue_mean > red_mean * 1.2:
        return img[..., ::-1].copy()

    return img


def _to_numpy_rgb(frame: ArrayLike) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))

    if torch.is_tensor(frame):
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3):
                arr = frame.detach().cpu()
                if arr.dtype.is_floating_point:
                    arr = (arr.clamp(0, 1) * 255).byte()
                else:
                    arr = arr.to(torch.uint8)
                arr = arr.permute(1, 2, 0).contiguous().numpy()
            else:
                arr = frame.detach().cpu().numpy()
                if arr.dtype.kind == "f":
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
        elif frame.ndim == 4:
            return _to_numpy_rgb(frame[0])
        else:
            raise ValueError("Unsupported tensor shape for frame.")
        return arr

    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    raise TypeError("frame must be np.ndarray, torch.Tensor, or PIL.Image")


class HumanDetector:
    
    def __init__(
        self,
        weights: str = "yolov8n-face.pt",
        device: Optional[Union[str, torch.device]] = None,
        conf: float = 0.30,
        iou: float = 0.45,
        half: bool = False,
        classes: Optional[Iterable[int]] = (0,),
        assume_bgr: bool = False
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.conf = float(conf)
        self.iou = float(iou)
        self.half = bool(half and self.device.type == "cuda")
        self.classes = None if classes is None else list(classes)
        self.assume_bgr = bool(assume_bgr)

        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = self._predict_internal(dummy)


    def predict(self, frame: ArrayLike) -> FrameDetections:
        frame_np = _to_numpy_rgb(frame)
        if isinstance(frame_np, np.ndarray):
            frame_np = auto_bgr_to_rgb(frame_np, force_assume_bgr=self.assume_bgr)
        return self._predict_internal(frame_np)

    def predict_batch(self, frames: List[ArrayLike]) -> List[FrameDetections]:
        prepared = []
        for f in frames:
            arr = _to_numpy_rgb(f)
            if isinstance(arr, np.ndarray):
                arr = auto_bgr_to_rgb(arr, force_assume_bgr=self.assume_bgr)
            prepared.append(arr)
        return self._predict_internal_batch(prepared)

    def select_primary_target(
        self,
        detections: FrameDetections,
        strategy: str = "largest_area",
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Det]:
        if not detections:
            return None
        if strategy == "highest_score":
            return max(detections, key=lambda d: float(d["score"]))

        def area(det):
            x1, y1, x2, y2 = det["bbox"].tolist()
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

        return max(detections, key=area)

    def _predict_internal(self, frame_rgb: np.ndarray) -> FrameDetections:
        results = self.model.predict(
            source=frame_rgb,
            verbose=False,
            device=str(self.device),
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
        )
        return self._results_to_detections(results)

    def _predict_internal_batch(self, frames_rgb: List[np.ndarray]) -> List[FrameDetections]:
        results = self.model.predict(
            source=frames_rgb,
            verbose=False,
            device=str(self.device),
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
        )
        out: List[FrameDetections] = []
        for r in results:
            out.append(self._results_to_detections([r]))
        return out

    @staticmethod
    def _results_to_detections(results) -> FrameDetections:
        r = results[0]
        dets: FrameDetections = []
        for b in r.boxes:
            xyxy = b.xyxy[0]
            score = float(b.conf.item())
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            dets.append({
                "bbox": xyxy.detach().cpu(),
                "score": score,
                "class_id": cls_id
            })
        return dets


#Model01 = HumanDetector(weights="yolov8n-face.pt")
#torch.save(Model01.model.state_dict(), "Model01.pt")