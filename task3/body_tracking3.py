import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import sys
import math  
import time


def compute_distance(position):
    """Calculate Euclidean distance from camera to person"""
    return math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)

def detect_hand_raise(body, image_scale):
    """
    Detect if a person is raising hands based on keypoint positions.
    Returns a string describing the hand raise status.
    """
    # Get 2D keypoints (normalized coordinates between 0 and 1)
    kps = body.keypoint_2d
    
    # Keypoint indices for BODY_18 format:
    # 0: Nose, 1: Neck, 2: Right Shoulder, 3: Right Elbow, 4: Right Wrist
    # 5: Left Shoulder, 6: Left Elbow, 7: Left Wrist
    # 8: Right Hip, 9: Right Knee, 10: Right Ankle
    # 11: Left Hip, 12: Left Knee, 13: Left Ankle
    # 14: Right Eye, 15: Left Eye, 16: Right Ear, 17: Left Ear
    
    right_shoulder = kps[2]
    right_wrist = kps[4]
    left_shoulder = kps[5]
    left_wrist = kps[7]
    
    # Check if wrist Y coordinate is above shoulder Y coordinate
    # Lower Y value means higher position in image (since origin is top-left)
    left_raised = False
    right_raised = False
    
    if left_wrist[1] * image_scale[1] < left_shoulder[1] * image_scale[1]:
        left_raised = True
    
    if right_wrist[1] * image_scale[1] < right_shoulder[1] * image_scale[1]:
        right_raised = True
    
    # Classify action
    if left_raised and right_raised:
        return "Both hands raised"
    elif left_raised:
        return "Left hand raised"
    elif right_raised:
        return "Right hand raised"
    else:
        return "No hand raised"

def parse_args(init):
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, default='HD720', help='Resolution')
    parser.add_argument('--model', type=str, default='MEDIUM', 
                       help='Body model: FAST, MEDIUM, ACCURATE')
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
    print("Running Body Tracking - Task 3: Hand Raise Detection with Distance")
    print("="*60)
    
    # Create ZED camera object
    zed = sl.Camera()
    
    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
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
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
    body_params.enable_tracking = True
    body_params.enable_segmentation = False
    body_params.enable_body_fitting = True
    body_params.body_format = sl.BODY_FORMAT.BODY_18  # Use 18-keypoint format
    
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
    
    # FPS initialization for reference
    prev_time = time.time()
    fps = 0.0
    
    print("Detection started. Press ESC to exit.")
    print("-"*40)
    
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Calculate FPS
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
            
            # Process each detected body
            if bodies.is_new:
                for body in bodies.body_list:
                    # Get 3D position
                    position = body.position
                    if position is not None and len(position) >= 3:
                        # Calculate distance
                        distance = compute_distance(position)
                        
                        # Detect hand raise action
                        action = detect_hand_raise(body, image_scale)
                        
                        # Format display text
                        text = f"{distance:.2f}m | {action}"
                        
                        # Get bounding box top-left corner for text placement
                        bounding_box = body.bounding_box_2d
                        if bounding_box is not None:
                            text_x = int(bounding_box[0][0] * image_scale[0])
                            text_y = int(bounding_box[0][1] * image_scale[1]) - 10
                            
                            # Draw background rectangle for text
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(image_left_ocv, 
                                        (text_x, text_y - text_h - 5),
                                        (text_x + text_w, text_y + 5),
                                        (0, 0, 0), -1)
                            
                            # Draw text
                            cv2.putText(image_left_ocv, text,
                                       (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Print to console for logging
                        print(f"Person: {text}")
                    
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
                    
                    # Highlight shoulders and wrists (keypoints for hand detection)
                    if len(body.keypoint_2d) > 7:
                        # Right shoulder (index 2)
                        rs_x = int(body.keypoint_2d[2][0] * image_scale[0])
                        rs_y = int(body.keypoint_2d[2][1] * image_scale[1])
                        cv2.circle(image_left_ocv, (rs_x, rs_y), 6, (255, 0, 0), -1)
                        
                        # Right wrist (index 4)
                        rw_x = int(body.keypoint_2d[4][0] * image_scale[0])
                        rw_y = int(body.keypoint_2d[4][1] * image_scale[1])
                        cv2.circle(image_left_ocv, (rw_x, rw_y), 6, (0, 255, 0), -1)
                        
                        # Left shoulder (index 5)
                        ls_x = int(body.keypoint_2d[5][0] * image_scale[0])
                        ls_y = int(body.keypoint_2d[5][1] * image_scale[1])
                        cv2.circle(image_left_ocv, (ls_x, ls_y), 6, (255, 0, 0), -1)
                        
                        # Left wrist (index 7)
                        lw_x = int(body.keypoint_2d[7][0] * image_scale[0])
                        lw_y = int(body.keypoint_2d[7][1] * image_scale[1])
                        cv2.circle(image_left_ocv, (lw_x, lw_y), 6, (0, 255, 0), -1)
            
            # Display FPS on image
            cv2.putText(image_left_ocv, f"FPS: {fps:.1f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(image_left_ocv, "Hand Raise Detection | ESC to exit", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Update viewer
            viewer.update_image(image_left_ocv)
            viewer.update_bodies(bodies)
    
    # Clean up
    zed.disable_body_tracking()
    zed.close()
    viewer.exit()
    cv2.destroyAllWindows()
    print("\nProgram terminated.")

if __name__ == '__main__':
    main()
