from NeuralNetwork import HumanDetector, auto_bgr_to_rgb, _to_numpy_rgb
from ImageProcessing import Camera
import cv2
import torch
import keyboard

Cam = Camera(device="cpu")

Model01 = HumanDetector(weights="yolov8n-face.pt")
Model01.model.load_state_dict(torch.load("Model01.pt"))
Model01.model.eval()

Cam_is_running = False
Last_frame = None

def Starting():
    Cam.start()

def Stoping():
    Cam.stop()

def Get1Frame():
    return Cam.get_frame()

print("press: q to quit, space to start and backspace to stop")

while True:

    if keyboard.is_pressed("q"):
        print("exiting...")
        Stoping()
        cv2.destroyAllWindows()
        break

    if keyboard.is_pressed("space") and not Cam_is_running:
        print("starting the model...")
        Starting()
        Cam_is_running = True
        print("you can get a single frame by pressing f")
    
    if keyboard.is_pressed("backspace") and Cam_is_running:
        print("stoping camera...")
        Stoping()
        Cam_is_running = False
    
    if Cam_is_running and Cam.cap is not None:

        Frame = Get1Frame()
        if Frame is None:
            continue

        Last_frame = Frame.copy()

        Detections = Model01.predict(Frame)

        for det in Detections:
            x1, y1, x2, y2 = map(int, det["bbox"].tolist())
            score = det["score"]
            cv2.rectangle(Frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(Frame, f"face {score:.2f}", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camera - Face Detection", Frame)
        cv2.waitKey(1)

        if keyboard.is_pressed("f") and Last_frame is not None:
            print("capturing frame...")
            dets = Model01.predict(Last_frame)
            for det in dets:
                print(f"  bbox={det['bbox'].tolist()}, score={det['score']:.2f}, class={det['class_id']}")