from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import cv2
from random import sample
from moviepy import *
import moviepy
import numpy as np
# from moviepy.editor import *
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
# Define the folder paths
input_folder = 'yolo_detections'  # Folder containing YOLO detection results
output_folder = 'scene_changes'   # Folder to save the results

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the input folder (assuming they're JSON files with YOLO detections)
file_names = os.listdir(input_folder)

for file_name in tqdm(file_names, desc='Processing detection files'):
    if file_name.endswith('.json'):  # Assuming YOLO outputs are saved as JSON
        # Read the JSON file
        with open(os.path.join(input_folder, file_name), 'r') as f:
            detection_data = json.load(f)
        
        # Initialize DataFrame to store frame data
        frames_data = []
        
        # Inspect the structure of the JSON data
        sorted_frames = sorted(detection_data.keys())
        
        # Iterate through frames
        for frame_index, frame_name in enumerate(sorted_frames):
            frame_data = detection_data[frame_name]
            # Check if each detection is a dictionary or if the entire structure is different
            # Check if object_detection exists and is not empty
            if "object_detection" in frame_data and frame_data["object_detection"]:
                # Find all person detections
                person_detections = [
                    detection for detection in frame_data["object_detection"] 
                    if detection["object"] == "person"
                ]
                
                # If we found person detections
                if person_detections:
                    # Use the detection with highest confidence
                    best_detection = max(person_detections, key=lambda x: x["confidence"])
                    bbox = best_detection["bbox"]
                    confidence = best_detection["confidence"]
                    
                    # Calculate centroid from bbox [x_min, y_min, x_max, y_max]
                    centroid_x = (bbox[0] + bbox[2]) / 2
                    centroid_y = (bbox[1] + bbox[3]) / 2
                    centroid = np.array([centroid_x, centroid_y])
                    
                    frames_data.append({
                        'frame_name': frame_name,
                        'frame_index': frame_index,
                        'centroid': centroid,
                        'confidence': confidence
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(frames_data)
        
        # If no frames had person detections, continue to next file
        if df.empty:
            print(f"No person detections found in {file_name}")
            continue

        # Rest of the code remains the same as before for calculating distances and velocities
        # Function to calculate Euclidean distance between two points
        def euclidean_distance(point1, point2):
            return np.linalg.norm(point2 - point1)
        
        # Initialize variables for distance calculation
        distances = []
        velocities = []
        prev_centroid = None
        prev_frame_index = None
        
        # Process each frame to calculate distances and velocities
        for _, row in df.iterrows():
            current_centroid = row['centroid']
            current_frame_index = row['frame_index']
            
            # Calculate distance if we have a previous centroid
            if prev_centroid is not None:
                # Calculate physical distance 
                distance = euclidean_distance(prev_centroid, current_centroid)
                distances.append(distance)
                
                # Calculate time difference (number of frames)
                # This assumes frames are evenly spaced in time
                time_diff = current_frame_index - prev_frame_index
                
                # Calculate velocity (distance/time)
                # If frames are dropped or missing, this accounts for the time gap
                velocity = distance / time_diff if time_diff > 0 else 0
                velocities.append(velocity)
            else:
                # For the first frame with detection, set distance and velocity to 0
                distances.append(0)
                velocities.append(0)
            
            # Update previous values
            prev_centroid = current_centroid
            prev_frame_index = current_frame_index
        
        # Add calculated values back to the DataFrame
        df['distance'] = distances
        df['velocity'] = velocities
        
        # Determine scene changes based on velocity threshold
        velocity_threshold = 65  # Adjust as needed for your videos
        df['scene_change'] = df['velocity'] > velocity_threshold
        
        # Extract the identifier from the filename
        video_id = os.path.splitext(file_name)[0]
        
        # Save full results
        result_path = os.path.join(output_folder, f'scene_changes_{video_id}.csv')
        df.to_csv(result_path, index=False)
        
        # Save scene change points separately
        scene_changes = df[df['scene_change'] == True]
        if not scene_changes.empty:
            scene_changes_path = os.path.join(output_folder, f'scene_change_points_{video_id}.csv')
            scene_changes.to_csv(scene_changes_path, index=False)

print(f"Results saved to '{output_folder}' folder")
## for creating a helpful timestamp in the frame names for quick look up in the video
def milliseconds_to_minutes_seconds(milliseconds):
    """Convert milliseconds to mm:ss format."""
    seconds = int(milliseconds / 1000)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}_{remaining_seconds:02d}"

def compute_mse(frame1, frame2):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = (gray_frame1.astype(np.float32) - gray_frame2.astype(np.float32)) ** 2
    mse = np.mean(diff)
    return mse

def detect_scene_changes(video_path, frame_location, threshold=500000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    prev_frame = None
    scene_changes = [0]

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert frame to grayscale for simplicity
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ###try converting to MSE instead of absolute different
        if prev_frame is not None:
            # Compute absolute difference between current and previous frame
            diff = cv2.absdiff(gray_frame, prev_frame)
            diff_sum = diff.sum()
            ####GETTING MEAN SQUARED ERROR
            # diff = gray_frame.astype(np.float32) - prev_frame.astype(np.float32) 
            # diff_sum = np.mean(diff ** 2)  

            if diff_sum > threshold:
                scene_changes.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        prev_frame = gray_frame
    cap.release()
    frame_location = "frames" 
    if not os.path.isdir(frame_location):
        os.makedirs(frame_location, exist_ok=True) 

        i = 0
        for timestamp in scene_changes:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                print("Failed to open video capture.")

            video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            ret, frame = video_capture.read()

            # Convert timestamp to mm_ss format
            time_str = milliseconds_to_minutes_seconds(timestamp)


            filename = f"f_{time_str}"
            output_dir = frame_location + '/' + filename + '_' + str(i) +'.jpg'
            if frame is not None:
                cv2.imwrite(output_dir, frame)
                print(f"Frame saved successfully as {output_dir}")
            else:
                print(f"Failed to extract frame at {time_str}")

            i+=1
