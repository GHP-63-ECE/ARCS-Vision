import cv2
import json
import time
import math
import gc
import torch
import threading
import requests
import numpy as np
import os
from datetime import datetime
import pandas as pd  # Leveraged for high-performance data structuring
from ultralytics import YOLO

# --- LOGGING SETUP (DIRECTORY, DATE, TIME, AND TEST NUMBER) ---
test_number = input("Enter Test Number (e.g., 1, 2, 3): ").strip()
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the primary outputs folder and a dedicated videos subfolder
log_folder = "terminal outputs"
video_folder = os.path.join(log_folder, "videos")
os.makedirs(video_folder, exist_ok=True)

# Build matching filenames for both the CSV log and the MP4 video recording
log_filename = f"test_{test_number}_{current_time}.csv"
log_filepath = os.path.join(log_folder, log_filename)

video_filename = f"test_{test_number}_{current_time}.mp4"
video_filepath = os.path.join(video_folder, video_filename)

print(f"\n=============================================")
print(f"🟢 PANDAS LOGGING ACTIVE: Saving CSV to  -> {log_filepath}")
print(f"🎬 VIDEO RECORDING ACTIVE: Saving MP4 to -> {video_filepath}")
print(f"=============================================\n")

# Optimized Pandas logging function to append rows dynamically without losing data on a crash
def log_to_csv(log_type, frame_id=None, target_count=None, payload="", current_fps=0.0):
    # 1. Handle Terminal Output Displays
    clean_payload = payload.strip()
    if log_type == "Mac Sent":
        print(f"[{current_fps:.1f} FPS] Mac Sent -> {clean_payload}")
    elif log_type == "ESP32 Echo":
        print(f"   └── 🎉 ESP32 Echo: {clean_payload}")
    else:
        print(clean_payload)

    # 2. Package data row into a Pandas DataFrame
    row_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]],
        "Log_Type": [log_type],
        "Frame_ID": [frame_id if frame_id is not None else ""],
        "Target_Count": [target_count if target_count is not None else ""],
        "Payload": [clean_payload]
    }
    df = pd.DataFrame(row_data)
    
    # 3. Append to CSV file. Write headers ONLY if the file is being freshly created.
    header_needed = not os.path.exists(log_filepath)
    df.to_csv(log_filepath, mode='a', index=False, header=header_needed)

# --- SERIALLY DRIVEN CONFIGURATION (MOVEMENT TEAM PIPELINE) ---
SERIAL_PORT = "/dev/cu.esp32devtest1" 
BAUD_RATE = 115200
arduino_serial = None

if SERIAL_PORT:
    log_to_csv("System", payload=f"Connecting to movement team's ESP32 on {SERIAL_PORT}...")
    try:
        import serial
        # Small timeout constraints ensure quick cycles without hanging the video stream
        arduino_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05, write_timeout=0.05)
        time.sleep(2)  # Allow ESP32 bootloader cycle to stabilize
        log_to_csv("System", payload="Serial communication pipeline established successfully!")
    except Exception as e:
        log_to_csv("System", payload=f"WARNING: Could not open serial port {SERIAL_PORT}. Running in visual-only mode. Error: {e}")

# --- OPTIMIZED, BREAK-PROOF ZERO-DELAY HTTP MJPEG PARSER ---
class RealTimeVideoStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            try:
                r = requests.get(self.url, stream=True, timeout=5, headers={'Connection': 'keep-alive'})
                bytes_data = bytes()
                for chunk in r.iter_content(chunk_size=4096):
                    if not self.running:
                        break
                    bytes_data += chunk
                    
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        
                        if len(jpg) > 0:
                            np_arr = np.frombuffer(jpg, dtype=np.uint8)
                            if np_arr.size > 0:
                                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                if frame is not None and frame.size > 0:
                                    with self.lock:
                                        self.frame = frame
                                        self.ret = True
            except Exception as e:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                img = self.frame.copy()
                self.frame = None 
                return self.ret, img
            return False, None

    def stop(self):
        self.running = False

# Initialize Model explicitly on the Mac GPU via MPS
weights_path = "/Users/daiwikgoel/Downloads/real data.yolov8/runs/segment/train/weights/best.pt"
log_to_csv("System", payload=f"Loading weights from: {weights_path}")

model = YOLO(weights_path)
log_to_csv("System", payload=f"M5 GPU Acceleration (MPS) Active: {torch.backends.mps.is_available()}")

# --- WIRELESS HTTP STREAM CONFIGURATION ---
PI_IP_ADDRESS = "100.108.166.82"  
MJPEG_URL = f"http://{PI_IP_ADDRESS}:8080/stream"

log_to_csv("System", payload=f"Connecting to wireless camera stream at: {MJPEG_URL}")
stream = RealTimeVideoStream(MJPEG_URL)

frame_id = 0
video_writer = None  # Placeholder object for the OpenCV recorder engine

# --- ADJUSTED CRACK FILTER SETTINGS ---
CONFIDENCE_THRESHOLD = 0.40   # Stable threshold for structural cracks
MIN_BOX_SIZE = 0.02           # Drop tiny pixel noise
BOX_GAP_TOLERANCE = 0.12      # Distance limit to unify segmented parts of the same crack

log_to_csv("System", payload="Autonomous zero-delay tracking engine online...")

try:
    while True:
        start_time = time.time()
        success, frame = stream.read()
        
        if not success or frame is None:
            time.sleep(0.001)
            continue

        fh, fw = frame.shape[:2]
        raw_detections = []
        filtered_detections = []

        # Run Fast Inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, device='mps', verbose=False)
        annotated_frame = frame.copy()
        
        for r in results:
            annotated_frame = r.plot(boxes=False, labels=False, conf=False)
            
            if r.masks is not None and r.boxes is not None:
                xyn_list = r.masks.xyn      
                xyxy_list = r.boxes.xyxy.cpu().numpy()
                conf_list = r.boxes.conf.cpu().numpy()
                
                for i in range(len(xyn_list)):
                    poly = xyn_list[i]
                    if len(poly) < 3: 
                        continue  
                    
                    ncx = float(np.mean(poly[:, 0]))
                    ncy = float(np.mean(poly[:, 1]))
                    
                    x1, y1, x2, y2 = xyxy_list[i]
                    conf = float(conf_list[i])
                    
                    nx1, ny1, nx2, ny2 = float(x1 / fw), float(y1 / fh), float(x2 / fw), float(y2 / fh)
                    box_w = nx2 - nx1
                    box_h = ny2 - ny1
                    
                    aspect_ratio = box_w / (box_h + 1e-6)
                    if box_w < MIN_BOX_SIZE and box_h < MIN_BOX_SIZE:
                        continue
                    if 0.80 < aspect_ratio < 1.25 and max(box_w, box_h) < 0.12:
                        continue  

                    raw_detections.append({
                        "cx": ncx, "cy": ncy,
                        "x1": nx1, "y1": ny1,
                        "x2": nx2, "y2": ny2,
                        "conf": conf
                    })

        # Structural Proximity Merger Logic
        if len(raw_detections) > 0:
            merged_groups = []
            for det in raw_detections:
                placed = False
                for group in merged_groups:
                    for member in group:
                        ix1 = max(det["x1"] - BOX_GAP_TOLERANCE, member["x1"])
                        iy1 = max(det["y1"] - BOX_GAP_TOLERANCE, member["y1"])
                        ix2 = min(det["x2"] + BOX_GAP_TOLERANCE, member["x2"])
                        iy2 = min(det["y2"] + BOX_GAP_TOLERANCE, member["y2"])
                        
                        if ix1 < ix2 and iy1 < iy2:  
                            group.append(det)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    merged_groups.append([det])

            for group in merged_groups:
                g_x1 = min([m["x1"] for m in group])
                g_y1 = min([m["y1"] for m in group])
                g_x2 = max([m["x2"] for m in group])
                g_y2 = max([m["y2"] for m in group])
                g_conf = max([m["conf"] for m in group])
                
                g_cx = float(np.mean([m["cx"] for m in group]))
                g_cy = float(np.mean([m["cy"] for m in group]))
                
                filtered_detections.append({
                    "cx": round(g_cx, 4), "cy": round(g_cy, 4),
                    "x1": round(g_x1, 4), "y1": round(g_y1, 4),
                    "x2": round(g_x2, 4), "y2": round(g_y2, 4),
                    "conf": round(g_conf, 4)
                })

        # Draw framework boxes and crosshairs
        for target in filtered_detections:
            p_x1, p_y1 = int(target["x1"] * fw), int(target["y1"] * fh)
            p_x2, p_y2 = int(target["x2"] * fw), int(target["y2"] * fh)
            pixel_cx, pixel_cy = int(target["cx"] * fw), int(target["cy"] * fh)
            
            cv2.rectangle(annotated_frame, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
            cv2.drawMarker(annotated_frame, (pixel_cx, pixel_cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 3)

        # --- VIDEO WRITER INITIALIZATION ---
        # We configure this when the first frame arrives so we know the precise size (fw, fh)
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Mac native optimized MP4 codec
            # We hardcode 20.0 FPS to match your Mac M5 target processing speed
            video_writer = cv2.VideoWriter(video_filepath, fourcc, 20.0, (fw, fh))

        # Record the current frame (complete with AI overlays) directly into the video file
        video_writer.write(annotated_frame)

        # Output Payload Packaging
        output_payload = {
            "f": frame_id,
            "n": len(filtered_detections),
            "d": filtered_detections if len(filtered_detections) > 0 else [{"cx": -1, "cy": -1, "x1": -1, "y1": -1, "x2": -1, "y2": -1, "conf": 0.0}]
        }

        out = json.dumps(output_payload) + "\n"
        fps = 1.0 / (time.time() - start_time)
        
        # Log outbound packet to CSV and Terminal
        log_to_csv("Mac Sent", frame_id=frame_id, target_count=len(filtered_detections), payload=out, current_fps=fps)
        
        # --- TRANSMIT DATA AND LISTEN FOR RESPONSES ---
        if arduino_serial and arduino_serial.is_open:
            try:
                # 1. Ship raw JSON line down the wire
                arduino_serial.write(out.encode('utf-8'))
                arduino_serial.flush()
                
                # 2. Check if the ESP32 threw it back, stripping buffer history to get the absolute newest match
                if arduino_serial.in_waiting > 0:
                    response = ""
                    while arduino_serial.in_waiting > 0:
                        response = arduino_serial.readline().decode('utf-8', errors='ignore').strip()
                    
                    if response and response.startswith("{") and response.endswith("}"):
                        # Log inbound echo straight to CSV and Terminal
                        log_to_csv("ESP32 Echo", frame_id=frame_id, target_count=len(filtered_detections), payload=response)
            except Exception as e:
                pass

        cv2.imshow("M5 Zero-Delay Receiver Window", annotated_frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_id % 120 == 0:
            torch.mps.empty_cache()
            gc.collect()

finally:
    stream.stop()
    
    # Safely seal and complete the MP4 video recording file
    if video_writer is not None:
        video_writer.release()
        log_to_csv("System", payload="Video recording saved and closed cleanly.")
        
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
        log_to_csv("System", payload="Serial pipeline disconnected safely.")
        
    cv2.destroyAllWindows()
