import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import sys
import time  

def parse_args(init):
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, default='HD720', help='Resolution (VGA, HD720, HD1200, HD2K)')
    args = parser.parse_args()
    if args.res == 'VGA':
        init.camera_resolution = sl.RESOLUTION.VGA
    elif args.res == 'HD2K':
        init.camera_resolution = sl.RESOLUTION.HD2K
    elif args.res == 'HD1200':
        init.camera_resolution = sl.RESOLUTION.HD1200
    else:
        init.camera_resolution = sl.RESOLUTION.HD720
    return args

def main():
    print("Running Body Tracking - Task 1: FPS Measurement")
    
    # Create ZED camera object
    zed = sl.Camera()
    
    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Change to VGA or HD2K as needed
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {err}")
        sys.exit(1)
    
    # Enable positional tracking 
    positional_params = sl.PositionalTrackingParameters()
    positional_params.set_floor_as_origin = True
    err = zed.enable_positional_tracking(positional_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable positional tracking: {err}")
        sys.exit(1)
    
    # Enable body tracking
    body_params = sl.BodyTrackingParameters()
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM  # Change to FAST or ACCURATE
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True
    
    err = zed.enable_body_tracking(body_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable body tracking: {err}")
        sys.exit(1)
    
    # Create OpenGL viewer
    viewer = sl.GLViewer()
    viewer.init()
    
    # Create image container
    image = sl.Mat()
    image_scale = [1, 1]
    
    # Initialize FPS variables
    prev_time = time.time()
    fps = 0.0
    
    # For power monitoring
    print("\n" + "="*60)
    print("FPS Measurement Started")
    print("="*60)
    print("For power monitoring, run 'jtop' in another terminal")
    print("Record VDD_IN values manually")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    
    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Compute FPS inside the loop
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            
            if dt > 0:
                fps = 1.0 / dt
            
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(1280, 720))
            image_left_ocv = image.get_data()
            
            # Retrieve bodies
            bodies = sl.Bodies()
            runtime_params = sl.BodyTrackingRuntimeParameters()
            runtime_params.detection_confidence_threshold = 40
            
            zed.retrieve_bodies(bodies, runtime_params)
            
            # Draw bodies
            if bodies.is_new:
                for body in bodies.body_list:
                    # Draw bounding box
                    bounding_box = body.bounding_box_2d
                    if bounding_box is not None:
                        p1 = (int(bounding_box[0][0] * image_scale[0]), 
                              int(bounding_box[0][1] * image_scale[1]))
                        p2 = (int(bounding_box[2][0] * image_scale[0]), 
                              int(bounding_box[2][1] * image_scale[1]))
                        cv2.rectangle(image_left_ocv, p1, p2, (0, 255, 0), 2)
                    
                    # Draw keypoints
                    for keypoint in body.keypoint_2d:
                        x = int(keypoint[0] * image_scale[0])
                        y = int(keypoint[1] * image_scale[1])
                        cv2.circle(image_left_ocv, (x, y), 4, (0, 0, 255), -1)
            
            # Display FPS on the image
            cv2.putText(image_left_ocv, f"FPS: {fps:.2f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Also display resolution and model info
            cv2.putText(image_left_ocv, f"Model: MEDIUM", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Update viewer
            viewer.update_image(image_left_ocv)
            viewer.update_bodies(bodies)
    
    # Calculate average FPS
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\nAverage FPS: {avg_fps:.2f}")
    
    # Clean up
    zed.disable_body_tracking()
    zed.close()
    viewer.exit()

if __name__ == '__main__':
    main()
