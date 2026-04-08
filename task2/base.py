import cv2
from ultralytics import YOLO
import pyzed.sl as sl
import time

# Load YOLO model on CPU
model = YOLO("yolov8n.pt")

# Open ZED camera
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTHMODE.NONE

status = zed.open(init)
if status != sl.ERRORCODE.SUCCESS:
    print("Failed to open ZED:", status)
    exit(1)

image = sl.Mat()
display_resolution = sl.Resolution(1280, 720)

# FPS initialization
prev_time = time.time()
fps = 0.0

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # FPS calculation
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        if dt > 0:
            fps = 1.0 / dt
        
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        frame = image.get_data()
        
        # Convert image format if needed
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Run YOLO on CPU
        results = model.predict(frame, device="cpu", verbose=False)
        annotated_frame = results[0].plot()
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20,40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("YOLO-CPU-on-ZED", annotated_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()
