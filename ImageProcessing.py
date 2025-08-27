import cv2
import torch

class Camera:
    def __init__(self, camera_index=0, device="cpu"):

        self.cap = None
        self.camera_index = camera_index
        self.device = device

    def start(self):

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError("Error: Could not open camera.")
        
        return self

    def stop(self):
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self, as_tensor=False):

        if self.cap is None:
            raise RuntimeError("Camera is not started. Call start() first.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame from camera.")

        if as_tensor:
            frame_tensor = torch.from_numpy(frame).to(self.device)
            frame_tensor = frame_tensor.permute(2, 0, 1)
            return frame_tensor

        return frame