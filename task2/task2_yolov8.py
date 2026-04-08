import cv2
from ultralytics import YOLO
import pyzed.sl as sl
import time
import numpy as np

def main():
    print("Running YOLOv8 CPU Object Detection")
    print("="*60)
    
    model = YOLO("yolov8n.pt")
    
    # Force CPU 
    model.to('cpu')
    print("Model loaded on CPU\n")
    
    # Configure and open ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth to save resources
    init_params.coordinate_units = sl.UNIT.METER
    
    print("Opening ZED camera...")
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)
    
    # Create image container
    image = sl.Mat()
    display_resolution = sl.Resolution(1280, 720)
    
    # FPS initialization
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    
    print("Starting object detection...")
    print("Press 'q' to quit\n")
    print("-"*40)
    
    while True:
        # Grab frame
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # FPS calculation
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            
            if dt > 0:
                fps = 1.0 / dt
                frame_count += 1
            
            # Retrieve image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            frame = image.get_data()
            
            # Convert RGBA to BGR for OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Run YOLO inference on CPU
            results = model.predict(frame, device="cpu", verbose=False, imgsz=640)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display info
            cv2.putText(annotated_frame, "Device: CPU | Model: YOLOv8n", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show number of detections
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(annotated_frame, f"Detections: {num_detections}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show window
            cv2.imshow("YOLO-CPU-on-ZED", annotated_frame)
            
            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
    
    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print("\n" + "="*40)
    print(f"Summary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print("="*40)
    
    # Clean up
    zed.close()
    cv2.destroyAllWindows()
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
