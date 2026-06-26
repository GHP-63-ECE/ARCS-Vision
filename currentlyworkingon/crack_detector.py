import cv2
import numpy as np
import serial
import json
import time
import sys
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List



# Folder must contain model.ncnn.param and model.ncnn.bin
MODEL_PATH = Path(__file__).parent / "best_ncnn_model"
MODEL_IMGSZ = 256

CONF = 0.25
IOU  = 0.45


# JEVOIS A33 CAMERA SETTINGS — CHANGE THESE FOR MVP CAMERAS


CAMERA_W = 320       # JeVois A33 native width 
CAMERA_H = 240       # JeVois A33 native height
CAMERA_FPS = 30      # CHANGE FOR OTHER CAMERA

# Linux: JeVois always shows as /dev/video0 (first USB cam plugged in)
# Windows: integer index, usually 0. If wrong camera, try 1 or 2.
# Mac: integer index, usually 0.
OS = platform.system()
if OS == "Linux":
    CAMERA_DEV = "/dev/video0"   # CHANGE if another USB cam takes video0
elif OS == "Windows":
    CAMERA_DEV = 0               # CHANGE index if JeVois isn't first camera
else:
    CAMERA_DEV = 0               # Mac — CHANGE index if needed


# SERIAL OUTPUT — where coordinates go (to model team)

if OS == "Linux":
    SERIAL_PORT = "/dev/ttyUSB0"    # USB-serial adapter
    # SERIAL_PORT = "/dev/ttyAMA0"  # if using RPi GPIO pins 8/10 instead
elif OS == "Windows":
    SERIAL_PORT = "COM3"            # CHANGE — check Device Manager
else:
    SERIAL_PORT = "/dev/tty.usbserial-0001"  # CHANGE — check ls /dev/tty.*

SERIAL_BAUD = 115200


# CAMERA GEOMETRY — not used in coord math yet, here for the other team
# Fill these in once you know the physical setup

CAMERA_HEIGHT_M = 1.0      # meters above the surface being inspected — CHANGE
CAMERA_TILT_DEG = 0.0      # degrees from vertical (0 = straight down) — CHANGE
CAMERA_FOV_H_DEG = 65.0    # JeVois A33 horizontal FOV — CHANGE FOR OTHER CAMERA




@dataclass
class Crack:
    id: int
    conf: float
    # normalised [0,1] relative to frame — (0,0) = top-left, (1,1) = bottom-right
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    area_frac: float   # fraction of total frame area covered by mask


def load_model():
    from ultralytics import YOLO

    param = MODEL_PATH / "model.ncnn.param"
    if not param.exists():
        # fall back to .pt if ncnn not exported yet (dev/testing)
        pt_files = list(MODEL_PATH.parent.glob("*.pt"))
        if not pt_files:
            print(f"No model found at {MODEL_PATH}")
            print("Expected folder: best_ncnn_model/  with model.ncnn.param + model.ncnn.bin")
            sys.exit(1)
        print(f"WARNING: NCNN not found, using {pt_files[0]} (slower)")
        return YOLO(str(pt_files[0]), task="segment")

    return YOLO(str(MODEL_PATH), task="segment")


def open_camera():
    if OS == "Windows":
        cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_DEV)

    if not cap.isOpened():
        print(f"Cannot open camera: {CAMERA_DEV}")
        print("Linux: run   ls /dev/video*   to list cameras")
        print("Windows: check Device Manager for camera index")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    # JeVois sends YUYV by default — request MJPG for better USB bandwidth
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera open: {actual_w}x{actual_h}  device={CAMERA_DEV}")
    return cap


def open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
        print(f"Serial open: {SERIAL_PORT} @ {SERIAL_BAUD}")
        return ser
    except Exception as e:
        print(f"Serial failed: {e}")
        print("Running without serial output — coords printed to terminal only")
        return None


def extract(result, fw, fh) -> List[Crack]:
    cracks = []
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return cracks

    frame_px = fw * fh

    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # default centroid from bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area_px = 0

        if masks is not None and i < len(masks):
            m = masks.data[i].cpu().numpy().astype(np.uint8)
            if m.shape != (fh, fw):
                m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
            area_px = int(np.sum(m))
            M = cv2.moments(m)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

        cracks.append(Crack(
            id        = i,
            conf      = round(conf, 3),
            cx        = round(cx / fw, 4),
            cy        = round(cy / fh, 4),
            x1        = round(x1 / fw, 4),
            y1        = round(y1 / fh, 4),
            x2        = round(x2 / fw, 4),
            y2        = round(y2 / fh, 4),
            area_frac = round(area_px / frame_px, 5) if frame_px > 0 else 0.0,
        ))

    return cracks


def send(ser, frame_id, cracks: List[Crack]):
    payload = {
        "f": frame_id,
        "n": len(cracks),
        "d": [asdict(c) for c in cracks],
    }
    line = (json.dumps(payload) + "\n").encode()

    if ser:
        try:
            ser.write(line)
        except Exception as e:
            print(f"Serial write error: {e}")
    else:
        print(line.decode(), end="")


def main():
    print(f"Platform: {OS}")
    print(f"Model: {MODEL_PATH}  imgsz={MODEL_IMGSZ}")

    model  = load_model()
    cap    = open_camera()
    ser    = open_serial()

    frame_id = 0
    t_last   = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed")
                time.sleep(0.05)
                continue

            fh, fw = frame.shape[:2]

            results = model.predict(
                source  = frame,
                imgsz   = MODEL_IMGSZ,
                conf    = CONF,
                iou     = IOU,
                verbose = False,
            )

            cracks = extract(results[0], fw, fh)
            send(ser, frame_id, cracks)

            # fps print every 60 frames
            if frame_id % 60 == 0:
                now = time.time()
                fps = 60 / (now - t_last) if frame_id > 0 else 0
                t_last = now
                print(f"frame={frame_id}  fps={fps:.1f}  cracks={len(cracks)}")

            frame_id += 1

    except KeyboardInterrupt:
        print("Stopped")
    finally:
        cap.release()
        if ser:
            ser.close()


if __name__ == "__main__":
    main()
