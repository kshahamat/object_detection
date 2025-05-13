from visual_features import VisualFeatures
import cv2
import argparse
import os
import numpy as np
from face_detection_er import *
import json
from scene_changes_time import detect_scene_changes
from transcript_output_v2 import process_audio_folder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='video.mp4')
    parser.add_argument('--frame_interval', type=int, default=20)
    parser.add_argument('--frame_location', type=str, default='./frames')
    parser.add_argument('--speech_seg', type=bool, default=True)
    parser.add_argument('--weapons', type=bool, default=False) #####SWITCH BACK TO TRUE TO RUN
    parser.add_argument('--ocr', type=bool, default=False) #####SWITCH BACK TO TRUE TO RUN
    parser.add_argument('--objects', type=bool, default=False)#####SWITCH BACK TO TRUE TO RUN
    parser.add_argument('--save_json', type=str, default='save_file.json')
    parser.add_argument('--emotion_recognition', type=bool, default=True)

    args = parser.parse_args()

    ## calls the detect change function to analyze the video from path and the frames from frame_location
    video_parser = detect_scene_changes(args.video_path,args.frame_location)
    #video_parser.frame_extraction(args.frame_interval,args.frame_location)

    ## creates a visual features object, checks that frames folder exists and extracts the frames from the command line input
    # (uses ./frame folder as default)
    extractor = VisualFeatures()
    os.makedirs('./frames', exist_ok=True)
    frames = os.listdir(args.frame_location)

    #dictionary for storing the frames
    features = {}

    def convert_ndarray(obj):
        """Convert NumPy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    if args.speech_seg:
        print("🔊 Running speech segmentation...")
        audio_folder = "./amz"  # or use another path if needed
        api_key = "e521205915e24617a14a74bf79424798"  # Ideally get from env or config
        process_audio_folder(audio_folder, api_key)

    ## Loop through each extracted frame
    # improved to indicate which features are being detected
    # Loop through each frame
    for frame in frames[:50]:
        frame_path = os.path.join(args.frame_location, frame)
        
        # Print frame path separately and add spacing
        print("\n" + "=" * 50)
        print(f"Frame Path: {frame_path}\n")
        
        if frame != 'video':  
            img = cv2.imread(frame_path)
            print("after cv2 imread")
            
            # Initialize a dictionary to store detected features for this frame
            features[frame] = {}

            # Extract and print features based on command-line arguments
            if args.weapons:
                output = extractor.weapon_detection(img)
                features[frame]['weapon_detection'] = output.tolist() if isinstance(output, np.ndarray) else output
                print("Weapon Detection:", output, "\n")

            if args.ocr:
                output = extractor.text_ocr(img)
                features[frame]['text_ocr'] = output.tolist() if isinstance(output, np.ndarray) else output
                print("Text OCR:", output, "\n")

            if args.objects:
                output = extractor.object_detection(img)
                features[frame]['object_detection'] = output.tolist() if isinstance(output, np.ndarray) else output
                print("Object Detection:", output, "\n")

            if args.emotion_recognition:
                output = extractor.emotion_recognition(img)
                # Ensure consistency in output format (list)
                if isinstance(output, int):
                    output = [output]
                    features[frame]['emotion_recognition'] = output.tolist() if isinstance(output, np.ndarray) else output
                print("Emotion Recognition:", output, "\n")

    # loop through each of the frames
    for frame in frames:
        print("Processing Frame:", frame)

        ## reads the frame using opencv to get the numpy array of the pixels
        if frame!='video':
          img = cv2.imread(args.frame_location+'/'+frame)
          # creates an empty list of the extracted features
          features[frame] = []
        
        # depending on the flags different features are extracted
        ## with functions from the visual features python file
        if args.weapons:
            output = extractor.weapon_detection(img)
            features[frame].append(output)
        if args.ocr:
              output = extractor.text_ocr(img)
              features[frame].append(output)
        if args.objects:
              output = extractor.object_detection(img)
              features[frame].append(output)
        
        if args.emotion_recognition:
            output = extractor.emotion_recognition(img)
            # if the output is a single integer it converts it to a list for consistency
            if isinstance(output, int):
                output = [output]

            features[frame].append(output)
    
    ## writes the feature dictionary in a new json file, default called save_file json
    with open(args.save_json, 'w') as f:
        json.dump(features, f, indent=4)

    print("\nFeature extraction completed. Results saved to", args.save_json)



